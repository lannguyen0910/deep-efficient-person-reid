from bisect import bisect_right
from datasets import *
from samplers import *
from nets import *
from engine import *
from augmentations import *
from utils.collate import *
from losses import *
from metrics import *
from configs import *

import torch

import torch.nn as nn
import torch.utils.data as data
import math

from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, LambdaLR, ReduceLROnPlateau, OneCycleLR, CosineAnnealingWarmRestarts
from utils.cuda import NativeScaler
from .random_seed import seed_everything
from .logger import setup_logger


def get_instance(config, **kwargs):
    # Inherited from https://github.com/vltanh/pytorch-template
    assert 'name' in config
    config.setdefault('args', {})
    if config['args'] is None:
        config['args'] = {}
    return globals()[config['name']](**config['args'], **kwargs)


def get_lr_policy(opt_config, model):
    optimizer_params = []

    if opt_config["name"] == 'sgd':
        optimizer_name = SGD
    elif opt_config["name"] == 'adam':
        optimizer_name = AdamW

    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        weight_decay = opt_config['weight_decay']
        lr = opt_config['lr']
        if "bias" in key:
            lr = opt_config['lr'] * opt_config['bias_lr_factor']
            weight_decay = opt_config['weight_decay_bias']

        if opt_config["name"] == 'sgd':
            optimizer_params += [{"params": [value], "lr": lr,
                                  "weight_decay": weight_decay,
                                  "momentum": opt_config['momentum'],
                                  "nesterov": True}]

        elif opt_config["name"] == 'adam':
            optimizer_params += [{"params": [value], "lr": lr,
                                  "weight_decay": weight_decay,
                                  'betas': (opt_config['momentum'], 0.999)}]

    optimizer = optimizer_name(optimizer_params)

    return optimizer


def get_lr_policy_with_center(opt_config, model, center_criterion):
    optimizer_params = []
    center_lr = opt_config['center_lr'] if 'center_lr' in opt_config.keys() \
        else None

    if opt_config["name"] == 'sgd':
        optimizer_name = SGD
    elif opt_config["name"] == 'adam':
        optimizer_name = AdamW

    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        weight_decay = opt_config['weight_decay']
        lr = opt_config['lr']

        if "bias" in key:
            lr = opt_config['lr'] * opt_config['bias_lr_factor']
            weight_decay = opt_config['weight_decay_bias']

        if opt_config["name"] == 'sgd':
            optimizer_params += [{"params": [value], "lr": lr,
                                  "weight_decay": weight_decay,
                                  "momentum": opt_config['momentum'],
                                  "nesterov": True}]

        elif opt_config["name"] == 'adam':
            optimizer_params += [{"params": [value], "lr": lr,
                                  "weight_decay": weight_decay,
                                  'betas': (opt_config['momentum'], 0.999), 'amsgrad': True}]

    optimizer = optimizer_name(optimizer_params)

    optimizer_center = AdamW(center_criterion.parameters(), lr=center_lr) if opt_config["name"] == 'adam' \
        else SGD(center_criterion.parameters(), lr=center_lr)

    return optimizer, optimizer_center

class EMA(torch.nn.Module):
    """
    [https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage]
    """

    def __init__(self, model, decay=0.9999):
        super().__init__()
        self.decay = decay
        self.model = copy.deepcopy(model).eval()

    def update(self, model):
        with torch.no_grad():
            e_std = self.model.state_dict().values()
            m_std = model.module.state_dict().values()
            for e, m in zip(e_std, m_std):
                e.copy_(self.decay * e + (1. - self.decay) * m)

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


def get_dataset_and_dataloader(config):

    train_transforms = build_transforms(config, is_train=True)
    val_transforms = build_transforms(config, is_train=False)

    dataset = init_dataset(config.dataset_names,
                           root=config.root_dir)

    num_classes = dataset.num_train_pids
    train_set = ImageDataset(dataset.train, train_transforms)

    if config.sampler == 'softmax':
        train_loader = DataLoader(
            train_set, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, collate_fn=train_collate_fn)
    else:
        train_loader = DataLoader(
            train_set, batch_size=config.batch_size,
            sampler=RandomIdentitySampler(
                dataset.train, config.batch_size, config.num_instance),
            # sampler=RandomIdentitySampler_alignedreid(dataset.train, config.num_instance),
            num_workers=config.num_workers, collate_fn=train_collate_fn
        )

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=config.batch_size * 2, shuffle=False, num_workers=config.num_workers,
        collate_fn=val_collate_fn
    )

    return train_loader, val_loader, len(dataset.query), num_classes

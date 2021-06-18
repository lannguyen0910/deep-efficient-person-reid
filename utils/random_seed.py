import numpy as np
import torch
import random
import os

SEED = 2021


def seed_everything(seed=SEED, device=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    if device:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.fastest = True
    else:
        torch.manual_seed(seed)

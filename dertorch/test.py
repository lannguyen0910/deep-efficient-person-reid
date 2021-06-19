from utils.getter import *
from os import mkdir

import argparse
import os
import sys
import torch
import gc
gc.enable()

sys.path.append('.')


def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]
                   ) if "WORLD_SIZE" in os.environ else 1

    config_path = os.path.join('configs', f'{args.config_file}.yaml')
    config = Config(config_path)

    output_dir = config.output_dir
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(config_path))
        with open(config_path, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(config))

    if config.device == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_devices

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.fastest = True

    train_loader, val_loader, num_query, num_classes = get_dataset_and_dataloader(
        config)
    model = Backbone(num_classes=num_classes, model_name=config.model_name)
    model.load_param(config.test_weight)

    inference(config, model, val_loader, num_query)

    torch.cuda.empty_cache()
    del model, train_loader, val_loader
    gc.collect()


if __name__ == '__main__':
    main()

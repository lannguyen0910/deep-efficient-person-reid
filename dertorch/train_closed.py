from utils.getter import *
import argparse
import os
import gc
import copy
gc.enable()

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True


def train(config):

    # prepare dataset
    train_loader, val_loader, _, _ = get_dataset_and_dataloader(
        config)
    num_query = config.num_query
    # prepare model
    # top_type 17 , bot_type 8, pid 2545

    # market
    # top_type 9, bot_type 10, pid 751
    num_classes_list = [config.num_classes, config.top_type, config.bot_type]
    model = ClosedBackbone(num_classes_list=num_classes_list, model_name=config.model_name, model_path=config.pretrain_path, pretrain_choice=config.pretrain_choice).to(config.device)



    if config.if_with_center == 'no':
        optimizer = get_lr_policy(config.lr_policy, model)
        scheduler = WarmupMultiStepLR(optimizer, config.steps, config.gamma, config.warmup_factor,
                                        config.warmup_iters, config.warmup_method)
        loss_func = make_loss_closed(config, num_classes_list)

        start_epoch = 0

        # Add for using self trained model
        # if config.pretrain_choice == 'self':
        #     start_epoch = eval(config.pretrain_path.split(
        #         '/')[-1].split('.')[0].split('_')[-1])
        #     print('Start epoch:', start_epoch)
        #     path_to_optimizer = config.pretrain_path.replace(
        #         'model', 'optimizer')
        #     print('Path to the checkpoint of optimizer:', path_to_optimizer)
        #     model.load_state_dict(torch.load(config.pretrain_path).state_dict())
        #     optimizer.load_state_dict(torch.load(path_to_optimizer).state_dict())
        #     scheduler = WarmupMultiStepLR(optimizer, config.steps, config.gamma, config.warmup_factor,
        #                                     config.warmup_iters, config.warmup_method, start_epoch)


        do_train_closed(
            config,
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,      # modify for using self trained model
            loss_func,
            num_query,
            start_epoch     # add for using self trained model
        )            
    else:
        loss_func, center_criterion = make_loss_closed(config, num_classes_list)
        optimizer, optimizer_center = get_lr_policy_with_center(
            config.lr_policy, model, center_criterion)
        scheduler = WarmupMultiStepLR(optimizer, config.steps, config.gamma, config.warmup_factor,
                                      config.warmup_iters, config.warmup_method)
        start_epoch = 0

        do_train_closed_with_center(
            config,
            model,
            center_criterion,
            train_loader,
            val_loader,
            optimizer,
            optimizer_center,
            scheduler,      # modify for using self trained model
            loss_func,
            num_query,
            start_epoch     # add for using self trained model
        )       

    torch.cuda.empty_cache()
    
    del model, optimizer, scheduler, train_loader, val_loader, loss_func
    gc.collect()                


def main():
    parser = argparse.ArgumentParser(description="Closed Dataset Training")
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
        os.makedirs(output_dir)

    logger = setup_logger("closed_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(config_path, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(config))

    if config.device == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_devices

    seed_everything(seed=config.seed)
    train(config)


if __name__ == '__main__':
    main()

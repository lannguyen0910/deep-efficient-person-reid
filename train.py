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
    train_loader, val_loader, num_query, num_classes = get_dataset_and_dataloader(
        config)

    # prepare model
    model = Backbone(num_classes=num_classes, model_name=config.model_name, model_path=config.pretrain_path, pretrain_choice=config.pretrain_choice).to(config.device)

    if config.if_with_center == 'no':
        print('Train without center loss, the loss type is',
              config.loss_type)
        optimizer = get_lr_policy(config.lr_policy, model)
        scheduler = WarmupMultiStepLR(optimizer, config.steps, config.gamma, config.warmup_factor,
                                      config.warmup_iters, config.warmup_method)
        loss_func = make_loss(config, num_classes)

        start_epoch = 0
        
        # Add for using self trained model
        if config.pretrain_choice == 'self':
            start_epoch = eval(config.pretrain_path.split(
                '/')[-1].split('.')[0].split('_')[-1])
            print('Start epoch:', start_epoch)
            path_to_optimizer = config.pretrain_path.replace(
                'model', 'optimizer')
            print('Path to the checkpoint of optimizer:', path_to_optimizer)
            model.load_state_dict(torch.load(config.pretrain_path).state_dict())
            optimizer.load_state_dict(torch.load(path_to_optimizer).state_dict())
            scheduler = WarmupMultiStepLR(optimizer, config.steps, config.gamma, config.warmup_factor,
                                          config.warmup_iters, config.warmup_method, start_epoch)


        do_train(
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

    elif config.if_with_center == 'yes':
        print('Train with center loss, the loss type is',
              config.loss_type)
        loss_func, center_criterion = make_loss_with_center(
            config, num_classes)
        optimizer, optimizer_center = get_lr_policy_with_center(
            config.lr_policy, model, center_criterion)
        scheduler = WarmupMultiStepLR(optimizer, config.steps, config.gamma, config.warmup_factor,
                                      config.warmup_iters, config.warmup_method)
        start_epoch = 0

        # Add for using self trained model
        if config.pretrain_choice == 'self':
            start_epoch = eval(config.pretrain_path.split(
                '/')[-1].split('.')[0].split('_')[-1])
            print('Start epoch:', start_epoch)
            path_to_optimizer = config.pretrain_path.replace(
                'model', 'optimizer')
            print('Path to the checkpoint of optimizer:', path_to_optimizer)
            path_to_center_param = config.pretrain_path.replace(
                'model', 'center_param')
            print('Path to the checkpoint of center_param:', path_to_center_param)
            path_to_optimizer_center = config.pretrain_path.replace(
                'model', 'optimizer_center')
            print('Path to the checkpoint of optimizer_center:',
                  path_to_optimizer_center)
            # print('Model state dict: ', model.state_dict())
            # print('model pretrain: ', torch.load(config.pretrain_path)._modules)
            model.load_state_dict(torch.load(config.pretrain_path).state_dict())
            optimizer.load_state_dict(torch.load(path_to_optimizer).state_dict())
            center_criterion.load_state_dict(torch.load(path_to_center_param).state_dict())
            optimizer_center.load_state_dict(
                torch.load(path_to_optimizer_center).state_dict())
            scheduler = WarmupMultiStepLR(optimizer, config.steps, config.gamma, config.warmup_factor,
                                          config.warmup_iters, config.warmup_method, start_epoch)

        do_train_with_center(
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
    else:
        print("Unsupported value for config.MODEL.IF_WITH_CENTER {}, only support yes or no!\n".format(
            config.if_with_center))

    torch.cuda.empty_cache()
    del model, optimizer, scheduler, train_loader, val_loader, loss_func, optimizer_center
    gc.collect()

def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]
                   ) if "WORLD_SIZE" in os.environ else 1

    config_path = os.path.join('configs', f'{args.config_file}.yaml')
    config = Config(config_path)

    # if args.config_file != "":
    #     config.merge_from_file(args.config_file)
    # config.merge_from_list(args.opts)
    # config.freeze()

    output_dir = config.output_dir
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
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


# def train(args, config):
#     os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_devices
#     num_gpus = len(config.gpu_devices.split(','))

#     device = torch.device(config.device)

#     trainloader, valloader, num_query, num_classes = get_dataset_and_dataloader(
#         config)

#     net = Backbone(num_classes=num_classes, model_name=config.model_name)

#     if args.saved_path is not None:
#         args.saved_path = os.path.join(args.saved_path, config.project_name)

#     if args.log_path is not None:
#         args.log_path = os.path.join(args.log_path, config.project_name)


#     optimizer, optimizer_params = get_lr_policy(config.lr_policy)

#     if config.mixed_precision:
#         scaler = NativeScaler()
#     else:
#         scaler = None

#     model = Classifier(
#         model=net,
#         metrics=metric,
#         scaler=scaler,
#         criterion=nn.CrossEntropyLoss(),
#         optimizer=optimizer,
#         optim_params=optimizer_params,
#         device=device)

#     if args.resume is not None:
#         load_checkpoint(model, args.resume)
#         start_epoch, start_iter, best_value = get_epoch_iters(args.resume)
#     else:
#         print('Not resume. Initialize weights')
#         start_epoch, start_iter, best_value = 0, 0, 0.0

#     scheduler, step_per_epoch = get_lr_scheduler(
#         model.optimizer,
#         lr_config=config.lr_scheduler,
#         num_epochs=config.num_epochs)

#     trainer = Trainer(config,
#                       model,
#                       trainloader,
#                       valloader,
#                       checkpoint=Checkpoint(
#                           save_per_iter=args.save_interval, path=args.saved_path),
#                       best_value=best_value,
#                       logger=Logger(log_dir=args.log_path),
#                       scheduler=scheduler,
#                       visualize_when_val=args.gradcam_visualization,
#                       evaluate_per_epoch=args.val_interval,
#                       step_per_epoch=step_per_epoch)

#     print("##########   DATASET INFO   ##########")
#     print("Trainset: ")
#     print(trainset)
#     print("Valset: ")
#     print(valset)
#     print()
#     print(trainer)
#     print()
#     print(config)
#     print(f'Training with {num_gpus} gpu(s)')
#     print(f"Start training at [{start_epoch}|{start_iter}]")
#     print(f"Current best MAP: {best_value}")

#     trainer.fit(start_epoch=start_epoch, start_iter=start_iter,
#                 num_epochs=config.num_epochs, print_per_iter=args.print_per_iter)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser('Training EfficientDet')
#     parser.add_argument('config', default='config', type=str,
#                         help='project file that contains parameters')
#     parser.add_argument('--print_per_iter', type=int,
#                         default=300, help='Number of iteration to print')
#     parser.add_argument('--val_interval', type=int, default=2,
#                         help='Number of epoches between valing phases')
#     parser.add_argument('--gradcam_visualization', action='store_true',
#                         help='whether to visualize box to ./sample when validating (for debug), default=off')
#     parser.add_argument('--save_interval', type=int,
#                         default=1000, help='Number of steps between saving')
#     parser.add_argument('--log_path', type=str, default='loggers/runs')
#     parser.add_argument('--resume', type=str, default=None,
#                         help='whether to load weights from a checkpoint, set None to initialize')
#     parser.add_argument('--saved_path', type=str, default='./weights')
#     parser.add_argument('--freeze_backbone', action='store_true',
#                         help='whether to freeze the backbone')

#     args = parser.parse_args()
#     config = Config(os.path.join('configs', 'configs.yaml'))

#     train(args, config)

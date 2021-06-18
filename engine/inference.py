import logging
import torch
import torch.nn as nn
import os
import numpy as np

from ignite.engine import Engine
from metrics.mAP import R1_mAP, R1_mAP_reranking


# def create_supervised_evaluator(model, metrics,
#                                 device=None):
#     """
#     Factory function for creating an evaluator for supervised models
#     Args:
#         model (`torch.nn.Module`): the model to train
#         metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
#         device (str, optional): device type specification (default: None).
#             Applies to both model and batches.
#     Returns:
#         Engine: an evaluator engine with supervised inference function
#     """
#     if device:
#         if torch.cuda.device_count() > 1:
#             model = nn.DataParallel(model)
#         model.to(device)

#     def _inference(engine, batch):
#         model.eval()
#         with torch.no_grad():
#             data, pids, camids = batch
#             data = data.to(device) if torch.cuda.device_count() >= 1 else data
#             feat = model(data)
#             return feat, pids, camids

#     engine = Engine(_inference)

#     for name, metric in metrics.items():
#         metric.attach(engine, name)

#     return engine

def inference(
        config,
        model,
        val_loader,
        num_query):
    device = config.device

    logger = logging.getLogger("reid_baseline.inference")
    logger.info("Enter inferencing")

    if config.test_reranking == 'no':
        evaluator = R1_mAP(num_query, max_rank=50,
                           feat_norm=config.test_feat_norm)
    else:
        evaluator = R1_mAP_reranking(num_query, max_rank=50,
                                     feat_norm=config.test_feat_norm)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(
                torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    print('Len val_loader: ', len(val_loader))
    for i, (img, pid, camid, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)

            if config.flip_feats == 'on':
                feat = torch.FloatTensor(img.size(0), 2048).zero_().cuda()
                for i in range(2):
                    if i == 1:
                        inv_idx = torch.arange(
                            img.size(3) - 1, -1, -1).long().cuda()
                        img = img.index_select(3, inv_idx)
                    f = model(img)
                    feat = feat + f
            else:
                feat = model(img)
            
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)
            print('Done update: ', i)

    cmc, mAP, distmat, pids, camids, qfeats, gfeats = evaluator.compute()
    print('Done compute evaluator!')
    
    if config.log_dir and not os.path.exists(config.log_dir):
        print('Create folder name: ', config.log_dir)
        os.makedirs(config.log_dir)

    np.save(os.path.join(config.log_dir, config.dist_mat), distmat)
    print('Saved distmat!')

    np.save(os.path.join(config.log_dir, config.pids), pids)
    print('Saved pids!')

    np.save(os.path.join(config.log_dir, config.camids), camids)
    print('Saved camids!')

    np.save(os.path.join(config.log_dir, config.img_path),
            img_path_list[num_query:])
    print('Saved img_path!')

    torch.save(qfeats, os.path.join(config.log_dir, config.qfeats))
    print('Saved qfeats!')

    torch.save(gfeats, os.path.join(config.log_dir, config.gfeats))
    print('Saved gfeats!')

    logger.info("Validation Results")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    # if config.test_reranking == 'no':
    #     print("Create evaluator")
    #     evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=config.test_feat_norm)},
    #                                             device=device)
    # elif config.test_reranking == 'yes':
    #     print("Create evaluator for reranking")
    #     evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP_reranking(num_query, max_rank=50, feat_norm=config.test_feat_norm)},
    #                                             device=device)
    # else:
    #     print("Unsupported re_ranking config. Only support for no or yes, but got {}.".format(
    #         config.test_reranking))

    # evaluator.run(val_loader)
    # cmc, mAP = evaluator.state.metrics['r1_mAP']
    # logger.info('Validation Results')
    # logger.info("mAP: {:.1%}".format(mAP))
    # for r in [1, 5, 10]:
    #     logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

import numpy as np
import torch
from tqdm import tqdm

# from orig code
def get_euclidean(x, y, **kwargs):
    m = x.shape[0]
    n = y.shape[0]
    distmat = (
        torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n)
        + torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    )
    distmat.addmm_(1, -2, x, y.t())
    return distmat




def get_cosine(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Computes cosine distance between two tensors.
    The cosine distance is the inverse cosine similarity
    -> cosine_distance = abs(-cosine_distance) to make it
    similar in behaviour to euclidean distance
    """
    sim_mt = cosine_similarity(x, y, eps)
    return torch.abs(1 - sim_mt).clamp(min=eps)


def _commpute_batches_double(qf, gf, dist_func="euclidean"):
    assert dist_func == "euclidean" or dist_func == "cosine"
    dist_func = get_euclidean if dist_func=="euclidean" else get_cosine
    gf_num = gf.shape[0]
    num_batches = (gf_num // 20000) + 35
    gf_batchsize = int((gf_num // num_batches))
    results = []

    if isinstance(qf, np.ndarray):
        qf = torch.from_numpy(qf).float().cuda()

    for i in tqdm(range(num_batches + 1)):
        
        gf_temp = gf[i * gf_batchsize : (i + 1) * gf_batchsize, :]
        gf_temp = torch.from_numpy(gf_temp).float().cuda()

        distmat_temp = dist_func(x=qf, y=gf_temp)
        results.append(distmat_temp.detach().cpu().numpy())
    
    return np.hstack(results)

def _commpute_batches_double_all(f, dist_func="euclidean"):
    assert dist_func == "euclidean" or dist_func == "cosine"
    dist_func = get_euclidean if dist_func=="euclidean" else get_cosine
    f_num = f.shape[0]
    num_batches = (f_num // 20000) + 35
    f_batchsize = int((f_num // num_batches))
    results = []

    # if isinstance(f, np.ndarray):
    #     f = torch.from_numpy(f).float().cuda()

    for i in tqdm(range(num_batches + 1)):
        results_column = []
        for j in range(num_batches + 1):
            lf_temp = f[i * f_batchsize : (i + 1) * f_batchsize, :]
            rf_temp = f[j * f_batchsize : (j + 1) * f_batchsize, :]
            lf_temp = lf_temp.float().cuda()
            rf_temp = rf_temp.float().cuda()

            distmat_temp = dist_func(x=lf_temp, y=rf_temp)
            results_column.append(distmat_temp.detach().cpu().numpy())

        print(np.shape(np.hstack(results_column)))
        results.append(np.hstack(results_column))
    print(np.shape(np.vstack(results)))
    return np.vstack(results)


def re_ranking(probFea, galFea, k1, k2, lambda_value, iter_batch=5000, dist_func='euclidean', local_distmat=None, only_local=False):
    # if feature vector is numpy, you should use 'torch.tensor' transform it to tensor
    query_num = probFea.size(0)
    all_num = query_num + galFea.size(0)
    assert dist_func == "euclidean" or dist_func == "cosine"
    dist_func = get_euclidean if dist_func=="euclidean" else get_cosine
    if only_local:
        original_dist = local_distmat
    else:
        feat = torch.cat([probFea, galFea])
        print('using GPU to compute original distance')
        original_dist = np.zeros(shape = [all_num,all_num],dtype = np.float16)
        i = 0
        while True:
            it = i + iter_batch
            print(it, np.shape(feat)[0])
            if it < np.shape(feat)[0]:
                original_dist[i:it,] = np.power(dist_func(feat[i:it,],feat).detach().cpu().numpy(),2).astype(np.float16)
            else:
                original_dist[i:,:] = np.power(dist_func(feat[i:,],feat).detach().cpu().numpy(),2).astype(np.float16)
                break
            i=it
        # distmat = _commpute_batches_double_all(feat)
        # distmat = torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num) + \
        #     torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(
        #         all_num, all_num).t()
        # distmat.addmm_(1, -2, feat, feat.t())
        # original_dist = distmat
        del feat
        if not local_distmat is None:
            original_dist = original_dist + local_distmat
    gallery_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    print('starting re_ranking')
    print(all_num, initial_rank.shape, k1, k2, lambda_value)
    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(
                np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                                            :int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(
                candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(
                    k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + \
        original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist

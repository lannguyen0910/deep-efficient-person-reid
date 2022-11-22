import torch


def train_collate_fn(batch):
    imgs, pids, _, _, = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids

def train_close_collate_fn(batch):
    imgs, pids, tops, bottoms = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    tops = torch.tensor(tops, dtype=torch.int64)
    bottoms = torch.tensor(bottoms, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, tops, bottoms
 
def val_collate_fn(batch):
    imgs, pids, camids, img_paths = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids, img_paths

def val_close_collate_fn(batch):
    imgs, pids, tops, bottoms = zip(*batch)
    return torch.stack(imgs, dim=0), pids, tops, bottoms
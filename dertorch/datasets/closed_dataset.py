import glob
import re
import os.path as osp
import pandas as pd
import json
from sklearn.model_selection import StratifiedKFold, train_test_split

from .base import BaseImageDataset


class ClosedDataset:
    """
    Dataset statistics:
    # identities: ?
    # images: 655603 (train) 
    """

    def __init__(self, root='innodep', verbose=True, **kwargs):
        super(ClosedDataset, self).__init__()
        # self.dataset_dir = osp.join(root, '1105_medium_relabel.pickle')

        self.train_dir = osp.join(root, '1117_train_relabel.pickle')
        self.val_dir = osp.join(root, '1117_val.pickle')

        self.split_dir = osp.join(root, '1027_split.pickle')


        # self.data_df = pd.read_pickle(self.dataset_dir)

        self.train_df = pd.read_pickle(self.train_dir)
        self.val_df = pd.read_pickle(self.val_dir)

        
        # TODO ; split pids with balanced
        # if not os.path.isfile(self.split_dir)
        #     data_df = self.data_df.reset_index()
        #     data_df = data_df.set_index(pd.Index(list(range(len(self.data_df))), name="index"))
        #     dataset_dict = data_df.to_dict("index")
        #     pids = np.array(data_df['pid'])
        #     train, test, _, _ = train_test_split(
        #         data_df, pids, test_size=0.1, stratify=pids
        #     )
        #     test_eids = [ds["index"] for ds in test_dataset]
        #     kfold = StratifiedKFold(n_splits=5)
        #     fold_eids = []
        #     labels = [ds['']]

        # now ; random split by (80%) 10% 10% / 3008 3344
        # medium ; 3028 3281
        # medium_relabel ; 657, 713
        # quantiles = [657, 713]
        # data_df = self.data_df.sample(frac=1)
        # print(data_df)
        # train = data_df[: int(len(data_df)*0.9)]
        # val = data_df[int(len(data_df)*0.9):]
        # test = data_df[int(len(data_df)*0.9): ]

        
        if verbose:
            print("=> closed dataset loaded")

        self.train = self.train_df.set_index(pd.Index(list(range(len(self.train_df))))).to_dict(
            "index"
        )
        self.val = self.val_df.set_index(pd.Index(list(range(len(self.val_df))))).to_dict(
            "index"
        )
        # self.test = test.set_index(pd.Index(list(range(len(test))))).to_dict(
        #     "index"
        # )

        # self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(
        #     self.train)
        # self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(
        #     self.query)
        # self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(
        #     self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError(
                "'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError(
                "'{}' is not available".format(self.gallery_dir))

        


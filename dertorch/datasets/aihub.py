import glob
import re
import os.path as osp

from .base import BaseImageDataset


class AIHUB(BaseImageDataset):

    def __init__(self, root='aihub_kor_dataset', verbose=True, **kwargs):
        super(AIHUB, self).__init__()

        self.dataset_dir = root
        self.train_dir = osp.join(self.dataset_dir, 'Training', 'training_image')
        self.query_dir = osp.join(self.dataset_dir, 'Validation', 'query_image')
        self.gallery_dir = osp.join(self.dataset_dir, 'Validation', 'gallery_image')

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Aihub dataset loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(
            self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(
            self.gallery)

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

    def _process_dir(self, dir_path, relabel=False):

        img_paths = glob.glob(osp.join(dir_path, '*.png'))



        # IN_H00997_SN3_110301_10650
        # H00997 110301
        # re?

        pattern = re.compile(r'(?:IN|OUT)_H([\d]+)_SN\d_([\d]+)_([\d]+)')

        pid_container = set()
        for img_path in img_paths:
            
            pid, _, _ = map(int, pattern.search(img_path).groups())
            # if pid == -1:
            #     continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}


        cid_container = set()
        for img_path in img_paths:
            
            _, cid, _ = map(int, pattern.search(img_path).groups())
            cid_container.add(cid)
        cid2label = {cid: label for label, cid in enumerate(cid_container)}

        dataset = []
        for img_path in img_paths:
            pid, cid, _ = map(int, pattern.search(img_path).groups())
            # if pid == -1:
            #     continue  # junk images are just ignored
            # assert 0 <= pid <= 938  # pid == 0 means background
            # assert 1 <= camid <= 6
            # camid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
                cid = cid2label[cid]
            dataset.append((img_path, pid, cid))

        return dataset

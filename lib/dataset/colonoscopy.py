import os
import pickle as pkl

import torch
import cv2
import numpy as np
from torch.utils.data import Dataset


class Colonoscopy(Dataset):

    @staticmethod
    def name2img_name(name):
        return name + '.jpg'

    @staticmethod
    def name2mask_name(name):
        return name + '_mask.jpg'

    def __init__(self, split_file, img_root, transforms):
        """

        :param split_file: split file for train test valid, each line is a image name
        :param img_root: img root path
        :param transforms:
        """
        self.img_root = img_root
        with open(split_file, 'r') as f:
            self.ids = [x.strip() for x in f.readlines()]
        self.ids = sorted(self.ids)
        self.transforms = transforms

    def __getitem__(self, index):
        """
        for densecrop that set dense_crop_cache=True, if cache exist, load from cache, otherwise read from file and dump
        :param index:
        :return:
        """
        name = self.ids[index]
        img_t, mask_t = self.get_transformed_img_mask(name)

        return img_t, mask_t, index

    def __len__(self):
        return len(self.ids)

    def get_transformed_img_mask(self, name):
        img_path = os.path.join(self.img_root, self.name2img_name(name))
        mask_path = os.path.join(self.img_root, self.name2mask_name(name))

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask[mask <= 127] = 0
        mask[mask > 127] = 1
        mask = mask.astype(np.uint8)

        img_t, mask_t = self.transforms(img, mask)

        return img_t, mask_t

    def get_img(self, index):
        """
        read image and mask, then transform
        :param index:
        :return:
        """
        name = self.ids[index]
        img_path = os.path.join(self.img_root, self.name2img_name(name))

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def get_mask(self, index):
        name = self.ids[index]
        mask_path = os.path.join(self.img_root, self.name2mask_name(name))

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask[mask <= 127] = 0
        mask[mask > 127] = 1
        mask = mask.astype(np.uint8)

        return mask

    def get_name(self, index):
        return self.ids[index]


def collate_fn(batch):
    """
    merge patches from multiple batches
    :param batch: list of list of ndarray, the first list is from different batch, second represent:
    [0]: transformed image, ndarray of shape N, H, W, C
    [1]: transformed mask, ndarray of shape N, H, W
    [2]: image index in dataset, int
    :return: batched result
    """

    batch_img = np.concatenate([b[0] for b in batch], axis=0)
    batch_mask = np.concatenate([b[1] for b in batch], axis=0)
    batch_index = [b[2] for b in batch]

    batch_img_t = torch.from_numpy(batch_img).permute(0, 3, 1, 2).contiguous().float()
    batch_mask_t = torch.from_numpy(batch_mask)

    return batch_img_t, batch_mask_t, batch_index

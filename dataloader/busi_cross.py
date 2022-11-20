import cv2
import json
import numpy as np
from os.path import join

import torch
from torch.utils.data import Dataset

from albumentations import *
import random

class busi_cross(Dataset):
    def __init__(self, args, mean, sd, flag='train'):
        self.args = args
        self.flag = flag
        self.mean = mean
        self.sd = sd
        self.imgs_dir = args.dir_img
        self.masks_dir = args.dir_mask
        self.img_size = args.img_size
        file_name = 'dataset'+ args.fold + '.json'
        self.fp = open(join('data', args.dataset, file_name), 'r')
        self.dataset = json.load(self.fp)
        self.fp.close()

        ##['test']这里要注意改到跟test.py的get_id一样才可以
        if self.flag == 'train':
           self.ids = self.dataset['train']
        elif self.flag == 'val':
            # self.ids = self.dataset['val']
            self.ids = self.dataset['test']
        elif self.flag == 'test':
            self.ids = self.dataset['test']

    def __len__(self):
        return len(self.ids)

    def preprocess(self, image, mask):

        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=2)

        if self.flag == 'train':
            aug = Compose([
                Resize(self.img_size, self.img_size),
                Rotate(limit=(-45, 45), p=0.5, border_mode=cv2.BORDER_CONSTANT),
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
            ])
        else:
            aug = Compose([
                Resize(self.img_size, self.img_size)])

        augmented = aug(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']

        self.mean = [0.485, 0.456, 0.406]
        self.sd = [0.229, 0.224, 0.225]

        if image.max() > 1:
            image = (image - self.mean) / self.sd
        if mask.max() > 1:
            mask = mask / 255.0

        # HWC to CHW
        image = image.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))
        image = image.astype(np.float32)
        mask = mask.astype(np.float32)
        return image, mask

    def __getitem__(self, i):
        idx = self.ids[i]
        img_file = join(self.imgs_dir, idx)
        mask_file = join(self.masks_dir, idx)

        img = cv2.imread(img_file)
        mask = cv2.imread(mask_file, 0)

        img1, mask1 = self.preprocess(img, mask)

        return {'image': torch.from_numpy(img1), 'mask': torch.from_numpy(mask1)}



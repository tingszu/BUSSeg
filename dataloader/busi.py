import cv2
import json
import numpy as np
from os.path import join

import torch
from torch.utils.data import Dataset

from albumentations import *
import random

class busi(Dataset):
    def __init__(self, args, mean, sd, flag='train'):
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

        if self.flag == 'train':
            # self.ids = self.dataset['trian']
            if args.fold == '':
                self.ids = self.dataset['unlabled'] + self.dataset['labled']
            else:
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

        # if self.flag == 'train':
        #     aug = Compose([
        #         Resize(self.img_size, self.img_size),
        #         Rotate(limit=(-45, 45), p=0.5),
        #         HorizontalFlip(p=0.5),
        #         VerticalFlip(p=0.5),
        #     ])
        # elif self.flag == 'val':
        #     aug = Compose([
        #         Resize(self.img_size, self.img_size),
        #         Rotate(limit=(-5, 5), p=0.5),
        #         HorizontalFlip(p=0.5),
        #         VerticalFlip(p=0.5),
        #     ])
        # else:
        #     aug = Compose([
        #         Resize(self.img_size, self.img_size)])

        if self.flag == 'train':
            aug = Compose([
                # OneOf([
                #     # RandomResizedCrop(height=224, width=224, scale=(0.5, 1), ratio=(0.6, 1), p=0.7),
                #     RandomResizedCrop(height=224, width=224, scale=(0.5, 1), ratio=(0.5, 1.8), p=0.5),
                # ]),
                Resize(self.img_size, self.img_size),
                # Rotate(limit=(-45, 45), p=0.5, border_mode=cv2.BORDER_CONSTANT),
                # HorizontalFlip(p=0.5),
                # VerticalFlip(p=0.5),
            ])
        else:
            aug = Compose([
                Resize(self.img_size, self.img_size)])

        augmented = aug(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']


        augmented = aug(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']

        self.mean = [0.485, 0.456, 0.406]
        self.sd = [0.229, 0.224, 0.225]

        if image.max() > 1:
            image = (image - self.mean) / self.sd
        # TODO
        # HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1:]
        # LAB = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        # image = np.concatenate((image, HSV, LAB), axis=-1)
        if mask.max() > 1:
            mask = mask / 255.0
            # mask[mask>1] = 1.0

        # HWC to CHW
        image = image.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))
        image = image.astype(np.float32)
        mask = mask.astype(np.float32)
        # edge = onehot_to_binary_edges(mask)
        # edge = edge.astype(np.float32)
        # return image, mask, edge
        return image, mask

    def __getitem__(self, i):
        idx = self.ids[i]
        img_file = join(self.imgs_dir, idx)
        mask_file = join(self.masks_dir, idx)

        if idx.startswith('normal'):
            label = [1, 0, 0]
            class_name = 'normal'
        elif idx.startswith('benign'):
            label = [0, 1, 0]
            class_name = 'benign'
        elif idx.startswith('malignant'):
            label = [0, 0, 1]
            class_name = 'malignant'

        # #cby code
        # img_file = self.ids[i]
        # mask_file = img_file.replace('image', 'mask')

        img = cv2.imread(img_file)
        # img = cv2.imread(img_file,cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_file, 0)

        # img, mask, edge = self.preprocess(img, mask)
        img, mask = self.preprocess(img, mask)

        # return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask), 'edge': torch.from_numpy(edge)}
        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask), 'label': torch.from_numpy(np.array(label))}



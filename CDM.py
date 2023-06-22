import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init, kaiming_init
from torch.nn.parameter import Parameter
import cv2
import numpy as np

def one_hot(masks):
    shp_x = (masks.size(0), 2, masks.size(2), masks.size(3))
    with torch.no_grad():
        masks = masks.long()
        y_onehot = torch.zeros(shp_x)
        if masks.device.type == "cuda":
            y_onehot = y_onehot.cuda(masks.device.index)
        y_onehot.scatter_(1, masks, 1)
    return y_onehot

def _construct_ideal_affinity_matrix(label1, label2):
    if label1.size()[1] == 1:
        label1 = one_hot(label1)
        label2 = one_hot(label2)
    label1 = label1.view(label1.size(0), label1.size(1), -1).float()
    label2 = label2.view(label2.size(0), label2.size(1), -1).float()
    ideal_affinity_matrix = torch.matmul(torch.transpose(label1, 1, 2).contiguous(), label2)
    return ideal_affinity_matrix


class CDM(nn.Module):
    def __init__(self, feat_channels=64, n_class = 2, bank_size=20):
        super().__init__()

        self.bank_size = bank_size

        self.mask = nn.Conv2d(feat_channels, n_class, kernel_size=1, padding=0)

        self.intra_conv = nn.Sequential(
            nn.BatchNorm2d(feat_channels),
            nn.Conv2d(feat_channels, feat_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True))
        self.inter_conv = nn.Sequential(
            nn.BatchNorm2d(feat_channels),
            nn.Conv2d(feat_channels, feat_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True))

        self.fuse = nn.Sequential(
            nn.Conv2d(feat_channels * 2, feat_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True))

        self.fuse_memory_conv = nn.Sequential(
            nn.Conv2d(feat_channels * (bank_size), feat_channels * 4, kernel_size=1, padding=0),
            nn.BatchNorm2d(feat_channels * 4),
            nn.ReLU(inplace=True))

        self.init_weight()

    def forward(self, bank, feature_ini, ptr, mask, mask_bank):

        batch_size = feature_ini.size()[0]

        relation_memory_list = []
        P_list = []
        affinity_matrix_list = []

        for i in range(self.bank_size):
            if i + batch_size <= self.bank_size:
                feature_bank = bank[i:i+batch_size]
                mask_bank_temp = mask_bank[i:i+batch_size]
            else:
                circle_ptr = batch_size - (self.bank_size - i)
                feature_bank = torch.cat([bank[i:self.bank_size], bank[0:circle_ptr]],dim=0)
                mask_bank_temp = torch.cat([mask_bank[i:self.bank_size], mask_bank[0:circle_ptr]],dim=0)

            feature_ = F.softmax(self.mask(feature_ini),dim=1)
            feature_bank_ =  F.softmax(self.mask(feature_bank),dim=1)

            fea_size = feature_ini.size()[2:] # h, W
            all_dim = fea_size[0] * fea_size[1] # b * c

            feature_bank_size = feature_bank.size()[2:]
            all_dim_bank = feature_bank_size[0] * feature_bank_size[1]

            feature_flat_ = feature_.view(-1, feature_.size()[1], all_dim) # [ b, c//4, h * w ]
            feature_bank_flat_ = feature_bank_.view(-1, feature_bank_.size()[1], all_dim_bank)

            feature_bank_flat = feature_bank.view(-1, feature_bank.size()[1], all_dim_bank)# [ b, c, h * w ]

            feature1_t_ = torch.transpose(feature_flat_, 1, 2).contiguous() # [ b, h * w, c ]

            A = torch.bmm(feature1_t_, feature_bank_flat_)
            P = F.softmax(A, dim=1)
            P_T = F.softmax(torch.transpose(A, 1, 2), dim=1)

            feature_intra = torch.bmm(feature_bank_flat, P_T).contiguous()
            feature_inter = torch.bmm(feature_bank_flat, 1-P_T).contiguous()

            feature_intra = feature_intra.view(-1, feature_ini.size()[1], fea_size[0], fea_size[1]) #相似特征增强后
            feature_inter = feature_inter.view(-1, feature_ini.size()[1], fea_size[0], fea_size[1]) #相似特征增强后

            feature_intra = self.intra_conv(feature_intra)
            feature_inter = self.inter_conv(feature_inter)

            feature_out = self.fuse(torch.cat([feature_intra, feature_inter], dim=1))

            affinity_matrix = _construct_ideal_affinity_matrix(mask, mask_bank_temp)

            # --append
            relation_memory_list.append(feature_out)
            P_list.append(P)
            affinity_matrix_list.append(affinity_matrix)

        # --concat
        feature_memory = torch.cat(relation_memory_list, dim=1)
        feature_memory = self.fuse_memory_conv(feature_memory)

        return feature_memory, P_list, affinity_matrix_list

    def init_weight(self):
            for ly in self.children():
                if isinstance(ly, nn.Conv2d):
                    nn.init.kaiming_normal_(ly.weight, a=1)
                    if not ly.bias is None: nn.init.constant_(ly.bias, 0)

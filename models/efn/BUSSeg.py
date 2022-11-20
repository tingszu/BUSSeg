import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from torch.nn import init
from .pvt import pvt_small, pvt_tiny, pvt_medium, PyramidVisionTransformer
from mmcv.cnn import constant_init, kaiming_init
from torch.nn.parameter import Parameter
from functools import partial
from torchvision import models
from .swin_transformer import SwinTransformer
import cv2
import numpy as np

class dsn(nn.Module):
    def __init__(self, ch_in, num_class=1):
        super(dsn, self).__init__()
        self.dsn = nn.Sequential(
            # nn.Dropout2d(0.2, False),
            nn.Conv2d(ch_in, ch_in//2, 1, bias=False),
            nn.BatchNorm2d(ch_in//2),
            nn.ReLU(False),
            nn.Conv2d(ch_in//2, num_class, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        return self.dsn(x)

class Efficient_Encoder(nn.Module):
    def __init__(self, efn, start, end):
        super(Efficient_Encoder, self).__init__()
        self.blocks = efn._blocks[start:end]
        self.blocks_len = len(efn._blocks)
        self.drop_connect_rate = efn._global_params.drop_connect_rate
        self.start = start
        self.end = end

    def forward(self, x):
        for idx, block in enumerate(self.blocks):
            drop_connect_rate = self.drop_connect_rate
            if self.drop_connect_rate:
                drop_connect_rate *= float(self.start + idx) / self.blocks_len
            x = block(x, drop_connect_rate=drop_connect_rate)
        return x

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # self.init_weight()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        return self.double_conv(x)

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, activation=None):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.activation = activation

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        if self.activation is not None:
            return self.activation(self.conv(x))
        return self.conv(x)

class efn_encoder(nn.Module):
    def __init__(self, model_name, filters, ends, n_channels=3, n_classes=1, bilinear=True,
                 activation=None, start_down=True):
        super(efn_encoder, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.start_down = start_down

        efn = EfficientNet.from_pretrained(model_name=model_name)

        if self.start_down:
            self.start_down = nn.Sequential(nn.Conv2d(n_channels, filters[0], kernel_size=7, stride=2, padding=3, bias=False),
                                            nn.BatchNorm2d(filters[0]),
                                            nn.ReLU(inplace=False))
        else:
            self.inc = DoubleConv(n_channels, filters[0])

        self.down1 = Efficient_Encoder(efn, start=0, end=ends[0])  # 24
        self.down2 = Efficient_Encoder(efn, start=ends[0], end=ends[1])  # 40
        self.down3 = Efficient_Encoder(efn, start=ends[1], end=ends[2])  # 80
        self.down4 = Efficient_Encoder(efn, start=ends[2], end=ends[3])  # 112

    def forward(self, x):

        if self.start_down:
            x1 = self.start_down(x)
        else:
            x1 = self.inc(x)

        x2 = self.down1(x1)  # 24
        x3 = self.down2(x2)  # 40
        x4 = self.down3(x3)  # 80
        x5 = self.down4(x4)  # 192

        return [x1, x2, x3, x4, x5]


class efn_decoder(nn.Module):
    def __init__(self, filters, n_channels=3, n_classes=1, bilinear=True,
                 activation=None):
        super(efn_decoder, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.up1 = Up(filters[4] + filters[3], filters[3], bilinear)
        self.up2 = Up(filters[3] + filters[2], filters[2], bilinear)
        self.up3 = Up(filters[2] + filters[1], filters[1], bilinear)
        self.up4 = Up(filters[1] + filters[0], filters[0], bilinear)
        self.outc = OutConv(filters[0], n_classes, activation)

    def forward(self, input,h=None):

        x = self.up1(input[-1], input[-2])
        x = self.up2(x, input[-3])
        x = self.up3(x, input[-4])
        x = self.up4(x, input[-5])

        feature_img_1 = None

        x = self.outc(x)
        return x,[feature_img_1]


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


class BUSSeg(nn.Module):
    def __init__(self, model_name='efficientnet-b0', n_channels=3, n_classes=3, bilinear=True,
                 activation=None, start_down=True, bank_size = 20, img_size = 384):
        super(BUSSeg, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.start_down = start_down

        efn_params = {
            'efficientnet-b0': {'filters': [32, 24, 40, 80, 192], 'ends': [3, 5, 8, 15]},
            'efficientnet-b1': {'filters': [32, 24, 40, 80, 192], 'ends': [5, 8, 12, 21]},
            'efficientnet-b2': {'filters': [32, 24, 48, 88, 208], 'ends': [5, 8, 12, 21]},
            'efficientnet-b3': {'filters': [40, 32, 48, 96, 232], 'ends': [5, 8, 13, 24]},
            'efficientnet-b4': {'filters': [48, 32, 56, 112, 272], 'ends': [6, 10, 16, 30]},
            'efficientnet-b5': {'filters': [48, 40, 64, 128, 304], 'ends': [8, 13, 20, 36]},
            'efficientnet-b6': {'filters': [56, 40, 72, 144, 344], 'ends': [9, 15, 23, 42]},
            'efficientnet-b7': {'filters': [64, 48, 80, 160, 384], 'ends': [11, 18, 28, 51]},
        }
        filters = efn_params[model_name]['filters']  # efn-b0的参数
        ends = efn_params[model_name]['ends']

        self.encoder = efn_encoder(model_name=model_name, filters = filters, ends = ends,
                                   start_down = start_down, n_channels=n_channels, n_classes=n_classes)

        self.transformer = pvt_tiny()
        pvt_filters = [64, 128, 320, 512]

        self.decoder = efn_decoder(filters=[filters[0],64, 128, 320, 512], bilinear=bilinear, n_channels=n_channels, n_classes=n_classes)

        self.neckblock5 = nn.Sequential(
            nn.Conv2d(pvt_filters[-1], pvt_filters[-1], kernel_size=1, padding=0),
            nn.BatchNorm2d(pvt_filters[-1]),
            nn.ReLU(inplace=True))
        self.neckblock4 = nn.Sequential(
            nn.Conv2d(pvt_filters[-2], pvt_filters[-2], kernel_size=1, padding=0),
            nn.BatchNorm2d(pvt_filters[-2]),
            nn.ReLU(inplace=True))
        self.neckblock3 = nn.Sequential(
            nn.Conv2d(pvt_filters[-3], pvt_filters[-3], kernel_size=1, padding=0),
            nn.BatchNorm2d(pvt_filters[-3]),
            nn.ReLU(inplace=True))
        self.neckblock2 = nn.Sequential(
            nn.Conv2d(pvt_filters[-4], pvt_filters[-4], kernel_size=1, padding=0),
            nn.BatchNorm2d(pvt_filters[-4]),
            nn.ReLU(inplace=True))

        self.fuse5 = nn.Sequential(
            nn.Conv2d(filters[-1], pvt_filters[-1], kernel_size=1, padding=0),
            nn.BatchNorm2d(pvt_filters[-1]),
            nn.ReLU(inplace=True))
        self.fuse4 = nn.Sequential(
            nn.Conv2d(filters[-2], pvt_filters[-2], kernel_size=1, padding=0),
            nn.BatchNorm2d(pvt_filters[-2]),
            nn.ReLU(inplace=True))
        self.fuse3 = nn.Sequential(
            nn.Conv2d(filters[-3], pvt_filters[-3], kernel_size=1, padding=0),
            nn.BatchNorm2d(pvt_filters[-3]),
            nn.ReLU(inplace=True))
        self.fuse2 = nn.Sequential(
            nn.Conv2d(filters[-4], pvt_filters[-4], kernel_size=1, padding=0),
            nn.BatchNorm2d(pvt_filters[-4]),
            nn.ReLU(inplace=True))

        self.bank_size = bank_size
        self.feat_channels = pvt_filters[-1] // 4

        self.bank_ptr = nn.Parameter(torch.zeros(1, dtype=torch.long), requires_grad=False)
        self.bank = nn.Parameter(torch.zeros(self.bank_size, self.feat_channels, img_size // 32, img_size // 32),requires_grad=False)
        self.bank_mask = nn.Parameter(torch.zeros(self.bank_size, self.feat_channels, img_size // 32, img_size // 32),requires_grad=False)
        self.bank_full = False

        self.neck_conv = nn.Sequential(
            nn.Conv2d(pvt_filters[-1], self.feat_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(self.feat_channels),
            nn.ReLU(inplace=False))

        self.CDM = CDM(feat_channels=self.feat_channels, n_class=2, bank_size=bank_size)

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(pvt_filters[-1] * 2, pvt_filters[-1], kernel_size=1, padding=0),
            nn.BatchNorm2d(pvt_filters[-1]),
            nn.ReLU(inplace=True))

    @torch.no_grad()
    def update_bank(self, x, mask):
        ptr = int(self.bank_ptr)
        batch_size = x.shape[0]
        vacancy = self.bank_size - ptr
        if batch_size >= vacancy:
            self.bank_full = True
        pos = min(batch_size, vacancy)
        self.bank[ptr:ptr + pos].data.copy_(x[0:pos].clone())
        self.bank_mask[ptr:ptr + pos].data.copy_(mask[0:pos].clone())
        # update pointer
        ptr = (ptr + pos) % self.bank_size
        self.bank_ptr[0] = ptr

    def forward(self, x1, mask, isMemory=True):

        b, c, h, w = x1.size()
        x_pvt_1 = self.transformer(x1)
        x1 = self.encoder(x1)
        x1[-1] = self.fuse5( x1[-1] ) + self.neckblock5(x_pvt_1[-1])
        x1[-2] = self.fuse4( x1[-2] ) + self.neckblock4(x_pvt_1[-2])
        x1[-3] = self.fuse3( x1[-3] ) + self.neckblock3(x_pvt_1[-3])
        x1[-4] = self.fuse2( x1[-4] ) + self.neckblock2(x_pvt_1[-4])

        #memory bank
        P_list = []
        affinity_matrix_list = []
        if isMemory:
            neck_feature = self.neck_conv(x1[-1])
            mask = F.interpolate(mask.float(), size=[neck_feature.size()[2], neck_feature.size()[3]], mode="nearest")
            ptr = int(self.bank_ptr)
            if self.bank_full == True:
                feature_memory, P_list, affinity_matrix_list = self.CDM(self.bank, neck_feature, ptr, mask, self.bank_mask)
                x1[-1] = self.fuse_conv(torch.cat([feature_memory, x1[-1]],dim=1))


        x1, feature_img_1 = self.decoder(x1,h)

        x = F.interpolate(x1, size=[h, w], mode="bilinear")

        return [x, P_list, affinity_matrix_list]

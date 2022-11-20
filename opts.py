from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import warnings
from distutils.version import LooseVersion

import os
import random
import numpy as np
from torch import nn

import torch
from torch import optim

from loss import *
from models import get_model

warnings.filterwarnings("ignore")
assert LooseVersion(torch.__version__) >= LooseVersion('1.3.0'), 'PyTorch>=1.3.0 is required'


def setup_seed(seed):
    """ 设置论文随机种子数相同 """
    if seed != 0:
        random.seed(seed)
        np.random.seed(seed)

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        os.environ['PYTHONHASHSEED'] = str(seed)


class Opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description=" Framework for 2D medical image segmentation",
                                              formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # Basic experiment setting
        self.parser.add_argument('-g', '--gpus', type=str, default='6',
                                 help='CUDA_VISIBLE_DEVICES')
        self.parser.add_argument('--exp_id', required=True, help='experiment name')
        self.parser.add_argument('--arch', required=True, help='model architecture.'
                                                               'UNet | *')
        self.parser.add_argument('--optim', default='Adam', help='optimizer alghorithm'
                                                                 'Adam | SGD | *')
        self.parser.add_argument('--sche', default='Poly', help='learning rate scheduler'
                                                                'Poly | ExpLR | MulStepLR | CosAnnLR | ReduceLR | *')
        self.parser.add_argument('--loss', default='dice_bce_loss', help='loss function.(default bce_loss)'
                                                                         'dice_loss | bce_loss | dice_bce_loss | *')
        self.parser.add_argument('--kth', type=int, default=-1,
                                 help='Index of k-fold.')
        # Model
        self.parser.add_argument('--img_size', type=int, default=224, help='size of image.')
        self.parser.add_argument('--n_channels', type=int, default=3, help='number of channels.')
        self.parser.add_argument('--n_classes', type=int, default=1, help='inumber of classes.')
        self.parser.add_argument('--ifCrossImage', action="store_true")
        self.parser.add_argument('--same', action="store_true")
        self.parser.add_argument('--cls', action="store_true")
        self.parser.add_argument('--ifSoftmax', action="store_true")
        # Train
        self.parser.add_argument('-e', '--epochs', type=int, default=100,
                                 help='Number of epochs')
        self.parser.add_argument('-b', '--batchsize', type=int, nargs='?', default=4,
                                 help='Batch size')
        self.parser.add_argument('-l', '--lr', type=float, nargs='?', default=1e-4,
                                 help='Learning rate')
        self.parser.add_argument('-s', '--seed', type=int, default=12345, help='random seed.')

        self.parser.add_argument('--resume', action='store_true', help='resume an experiment.')
        # Dataset
        self.parser.add_argument('--dataset', required=True, help='please specify the dataset which you use.')
        self.parser.add_argument('--fold', default='', help='please specify the dataset which you use.')
        # Test
        self.parser.add_argument('--tta', action='store_true', help='Test Time Augmentation.')
        self.parser.add_argument('--threshold', type=int, default=0,
                                 help='threshold of mask for post-process.')
        self.parser.add_argument('-p', '--pretrain', default='',
                                 help='pretrain model.')

    def parse_arg(self):
        opt = self.parser.parse_args()
        setup_seed(opt.seed)
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ####################################################################################################
        """ Directory """
        opt.dir_data = os.path.join('/data/hxt', opt.dataset)
        opt.dir_img = os.path.join(opt.dir_data, 'image')
        opt.dir_mask = os.path.join(opt.dir_data, 'mask')
        opt.dir_log = os.path.join(opt.dir_data, 'logs_data')
        opt.dir_test = os.path.join(opt.dir_data, 'test')
        opt.dir_vis = os.path.join('vis', opt.dataset, opt.exp_id)
        opt.dir_result = os.path.join('results', opt.dataset, f"EXP_{opt.exp_id}_NET_{opt.arch}")
        opt.dir_ckpt = os.path.join('logs', opt.dataset, f"EXP_{opt.exp_id}_NET_{opt.arch}")
        ####################################################################################################
        """ Dataset """
        if opt.dataset in ['thyroid', 'isic2018']:
            opt.n_test = len(list(os.listdir(opt.dir_test)))
        ####################################################################################################
        opt.net = get_model(3, opt.n_classes, opt.arch)
        if len(opt.gpus) > 1:
            opt.net = nn.DataParallel(opt.net)
        opt.net.to(device=opt.device)
        ####################################################################################################
        """ Optimizer """
        if opt.optim == "Adam":
            opt.optimizer = optim.Adam(opt.net.parameters(), lr=opt.lr, weight_decay=1e-5)
        elif opt.optim == "SGD":
            opt.optimizer = optim.SGD(opt.net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=1e-8)
        ####################################################################################################
        """ Scheduler """
        if opt.sche == "ExpLR":
            gamma = 0.95
            opt.scheduler = torch.optim.lr_scheduler.ExponentialLR(opt.optimizer, gamma=gamma, last_epoch=-1)
        elif opt.sche == "MulStepLR":
            milestones = [30, 90]
            opt.scheduler = torch.optim.lr_scheduler.MultiStepLR(opt.optimizer, milestones=milestones, gamma=0.1)
        elif opt.sche == "CosAnnLR":
            t_max = 5
            opt.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt.optimizer, T_max=t_max, eta_min=0.)
        elif opt.sche == "ReduceLR":
            mode = "min"
            factor = 0.9
            patience = 2
            opt.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt.optimizer, mode=mode, factor=factor,
                                                                       patience=patience)
        elif opt.sche == "StepLR":
            opt.scheduler = torch.optim.lr_scheduler.StepLR(opt.optimizer, step_size=2, gamma=0.9)
        ####################################################################################################
        """ Loss Function """
        if opt.loss == "dice_loss":
            opt.loss_function = DiceLoss()
        elif opt.loss == "dice_bce_loss":
            opt.loss_function = DiceBCELoss()
        elif opt.loss == "dice_jaccard_loss":
            opt.loss_function = DiceJaccardLoss()
        ####################################################################################################

        return opt

    def init(self):
        return self.parse_arg()

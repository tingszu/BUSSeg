import os
from tqdm import tqdm
from time import time
from runx.logx import logx
from datetime import datetime
from prettytable import PrettyTable
import torch.nn.functional as F

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from opts import Opts
from eval import eval_net
from dataloader import *
from utils.visualizer import Visualizer
from utils.preprocess import get_properties
from loss import DiceBCELoss, Criterion_co, Criterion_cross

def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs) ** exponent

def get_dataset(flag):
    shuffle = True if flag == 'train' else False
    batch_size = args.batchsize if flag == 'train' else 1
    if args.dataset == "kvasir":
        datasets = busi_cross(args, mean, std, flag=flag)
    elif args.dataset == "busi":
        if args.arch in ["Res_class", "Res_class_co", "Res_class_co2", 'Res_class_co_split']:
            datasets = busi_cross(args, mean, std, flag=flag)
        else:
            datasets = busi_cross(args, mean, std, flag=flag)
            # datasets = busi(args, mean, std, flag=flag)
    elif args.dataset == "datasetB":
        if args.arch in ["Res_class", "Res_class_co", "Res_class_co2", 'Res_class_co_split']:
            datasets = datasetB_cross(args, mean, std, flag=flag)
        else:
            datasets = datasetB_cross(args, mean, std, flag=flag)
    else:
        datasets = None
    data_loader = DataLoader(datasets, batch_size=batch_size,
                             shuffle=shuffle, num_workers=4,
                             pin_memory=True, drop_last=True)
    return data_loader, len(datasets)


def train_net():
    # Initialize the hyper-parameters
    start_epoch, global_step, best_score, min_loss = -1, 1, 0.0, 100.0
    viz = Visualizer(env=f"EXP_{args.exp_id}_NET_{args.arch}")

    if args.pretrain != '':
        checkpoint = torch.load(args.pretrain)
        args.net.load_state_dict(checkpoint["net"])

    # Resume the state if need
    if args.resume:
        checkpoint = torch.load(os.path.join(args.dir_ckpt, 'INTERRUPT.pth'))
        args.net.load_state_dict(checkpoint["net"])
        try:
            args.scheduler.load_state_dict(checkpoint["scheduler"])
        except AttributeError:
            pass
        args.optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint['epoch']

    for epoch in range(start_epoch + 1, args.epochs):
        epoch_start_time = time()
        epoch_loss, epoch_seg_loss, epoch_seg_loss1, epoch_seg_loss2, epoch_class_loss, epoch_class_loss1,epoch_class_loss2 = 0., 0. , 0., 0., 0., 0., 0.


        # train flag
        args.net.train()
        # get the current learning rate
        if args.sche == "Poly":
            new_lr = poly_lr(epoch, args.epochs + 1, args.lr, 0.9)
            args.optimizer.param_groups[0]['lr'] = new_lr
        else:
            new_lr = args.optimizer.state_dict()['param_groups'][0]['lr']


        # Train process
        with tqdm(total=args.n_train, desc=f'Epoch-{epoch + 1}/{args.epochs}', unit='img') as pbar:
            for batch in args.train_loader:

                imgs, true_masks  = batch['image'], batch['mask']
                assert imgs.shape[1] == args.n_channels

                # Prepare the image and the corresponding mask.
                imgs = imgs.to(device=args.device, dtype=torch.float32)
                mask_type = torch.float32 if args.n_classes == 1 else torch.long 
                true_masks = true_masks.to(device=args.device, dtype=mask_type)

                # Forward propagation.
                isMemory = False if epoch <= 10 else True
                if args.ifCrossImage:
                    masks_pred = args.net(imgs, true_masks, isMemory=isMemory)  
                    loss, seg_loss,  class_loss = criterion(masks_pred, true_masks, is_loss=True)
                    masks_pred = masks_pred[0] 
                else:
                    if args.arch == 'DCRNet':
                        masks_pred = args.net(imgs, flag='train')
                        loss = criterion(masks_pred, true_masks)
                    else:
                        masks_pred = args.net(imgs)
                        loss = criterion(masks_pred, true_masks)

                if isinstance(loss, list):
                    loss_temp = loss
                    loss, seg_loss, class_loss = loss_temp[0], loss_temp[1], loss_temp[2]
                    epoch_seg_loss += seg_loss.item()
                    epoch_class_loss += class_loss.item()

                logx.add_scalar('Loss/train', loss.item(), global_step)
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                if args.ifCrossImage:
                        epoch_seg_loss += seg_loss.item()
                        epoch_class_loss += class_loss.item()


                # Back propagation.
                args.optimizer.zero_grad()
                loss.backward()
                args.optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
        if isinstance(masks_pred, list):
            masks_pred = masks_pred[0]
            if isinstance(masks_pred, list):
                masks_pred = masks_pred[0]
        # Calculate  the train loss
        train_loss = epoch_loss / (args.n_train // args.batchsize)
        if args.ifCrossImage or args.arch.endswith('_scon'):
            if args.same:
                train_seg_loss1 = epoch_seg_loss1 / (args.n_train // args.batchsize)
                train_seg_loss2 = epoch_seg_loss2 / (args.n_train // args.batchsize)
                train_class_loss1 = epoch_class_loss1 / (args.n_train // args.batchsize)
                train_class_loss2 = epoch_class_loss2 / (args.n_train // args.batchsize)
                metrics = {'train_loss': train_loss, 'train_seg_loss1': train_seg_loss1, 'train_seg_loss2': train_seg_loss2,
                           'train_class_loss1': train_class_loss1, 'train_class_loss2': train_class_loss2}
            else :
                train_seg_loss = epoch_seg_loss / (args.n_train // args.batchsize)
                train_class_loss = epoch_class_loss / (args.n_train // args.batchsize)
                metrics = {'train_loss': train_loss, 'train_seg_loss': train_seg_loss,
                            'train_class_loss': train_class_loss}
        elif args:
            train_seg_loss = epoch_seg_loss / (args.n_train // args.batchsize)
            train_class_loss = epoch_class_loss / (args.n_train // args.batchsize)
            metrics = {'train_loss': train_loss, 'train_seg_loss': train_seg_loss,
                       'train_class_loss': train_class_loss}
        else:
            metrics = {'train_loss': train_loss}
        logx.metric(phase='train', metrics=metrics, epoch=epoch)

        # Validate process
        val_score, val_loss = eval_net(criterion, logx, epoch, args, viz, isMemory)

        # Update the current learning rate and
        # you should write the monitor metrics in step() if you use the ReduceLROnPlateau scheduler.
        if args.sche == "StepLR":
            args.scheduler.step()
        elif args.sche not in ["Poly"]:
            if isinstance(val_loss, list):
                args.scheduler.step(val_loss[1])
            else:
                args.scheduler.step(val_loss)

        # Caculating and logging the metrics
        save_by_metrics = 'dc'
        if args.ifCrossImage or args.cls or args.arch.endswith('_scon'):
            metrics = {
                'val_loss': val_loss[0],
                'seg_loss': val_loss[1],
                'class_loss': val_loss[2]
            }
        else:
            metrics = {
                'val_loss': val_loss
            }

        metrics['jc'] = val_score['jc']
        metrics['dc'] = val_score['dc']
        metrics['sp'] = val_score['sp']
        metrics['se'] = val_score['se']
        metrics['auc'] = val_score['auc']
        metrics['acc'] = val_score['acc']

        logx.metric(phase='val', metrics=metrics, epoch=epoch)

        # Print the metrics
        print("\033[1;33;44m=============================Evaluation result=============================\033[0m")
        if args.ifCrossImage or args.arch.endswith('_scon'):
            logx.msg("[Train] Loss: %.4f | Seg_Loss: %.4f | Class_Loss: %.4f | LR: %.6f" \
                         % (train_loss, train_seg_loss, train_class_loss, new_lr))
        else:
            logx.msg("[Train] Loss: %.4f | LR: %.6f" % (train_loss, new_lr))

        if args.ifCrossImage or args.cls or args.arch.endswith('_scon'):
            logx.msg("[Valid] Loss: %.4f | Seg_Loss: %.4f | Class_Loss: %.4f | ACC: %.4f | JC: %.4f | DC: %.4f" % (
            val_loss[0], val_loss[1], val_loss[2], metrics['acc'], metrics['jc'], metrics['dc'],))
        else:
            logx.msg("[Valid] Loss: %.4f | ACC: %.4f | JC: %.4f | DC: %.4f" % (
            val_loss, metrics['acc'], metrics['jc'], metrics['dc'],))

        # Logging the image for tensorboard
        logx.add_image('image', torch.cat([i for i in imgs], 2), epoch)
        logx.add_image('mask/gt', torch.cat([j for j in true_masks], 2), epoch)
        logx.add_image('mask/pd', torch.cat([k > 0.5 for k in masks_pred], 2), epoch)

        if args.ifCrossImage or args.cls or args.arch.endswith('_scon'):
            # val_loss = val_loss[0] 
            val_loss = val_loss[1]

        # Update the best score
        if best_score <  metrics['dc']:
            torch.save(args.net.state_dict(), f'{args.dir_ckpt}/{args.dataset}_{args.arch}_{args.exp_id}_best_val.pth')
            logx.msg("val_score improved from {:.4f} to {:.4f} "
                     "and the epoch {} took {:.2f}s.\n".
                     format(best_score,  metrics['dc'], epoch + 1, time() - epoch_start_time))
            best_score =  metrics['dc']
        if min_loss > val_loss :
            torch.save(args.net.state_dict(), f'{args.dir_ckpt}/{args.dataset}_{args.arch}_{args.exp_id}.pth')
            logx.msg("val_loss reduced from {:.4f} to {:.4f} "
                     "and the epoch {} took {:.2f}s.\n".
                     format(min_loss, val_loss, epoch + 1, time() - epoch_start_time))
            min_loss = val_loss
        else:
            logx.msg("val_score did not improved from {:.4f} "
                     "and the epoch {} took {:.2f}s.\n".
                     format(best_score, epoch + 1, time() - epoch_start_time))

        # Saving the model with relevant parameters
        if epoch % 3 == 0:
            checkpoint = {
                "net": args.net.state_dict(),
                "optimizer": args.optimizer.state_dict(),
                "scheduler": args.scheduler.state_dict() if args.sche != "Poly" else new_lr,
                "epoch": epoch + 1,
            }
            torch.save(checkpoint, os.path.join(args.dir_ckpt, 'INTERRUPT.pth'))


if __name__ == '__main__':
    args = Opts().init()

    # Data loader
    results = get_properties(args=args)
    mean, std = results['mean'], results['sd']
    args.train_loader, args.n_train = get_dataset(flag='train')
    args.val_loader, args.n_val = get_dataset(flag='val')

    # criterion = DiceBCELoss()
    # criterion = CriterionOhemDSN()
    # if args.arch in ["Res_class", "Res_class_co", "Res_class_co2", 'Res_class_co_split']:
    if args.ifCrossImage:
        criterion = Criterion_co()
    elif args.arch in ["UNet_Res_cross"]:
        criterion = Criterion_cross()
    elif args.ifSoftmax:
        criterion = args.loss_function
    else:
        criterion = nn.CrossEntropyLoss() if args.n_classes > 1 else args.loss_function

    # Initialize the information
    logx.initialize(logdir=args.dir_ckpt, coolname=True, tensorboard=True)
    logx.msg('Start training...\n')
    table = PrettyTable(["key", "value", ])
    table.align = 'l'
    table.add_row(['Epochs', f'{args.epochs}'])
    table.add_row(['Batch size', f'{args.batchsize}'])
    table.add_row(['Loss function', f'{args.loss}'])
    table.add_row(['Optimizer', f'{args.optim}'])
    table.add_row(['Random seed', f'{args.seed}'])
    table.add_row(['Architecture', f'{args.arch}'])
    table.add_row(['Dataset', f'{args.dataset}'])
    table.add_row(['Experiment id', f'{args.exp_id}'])
    table.add_row(['Fold id', f'{args.fold}'])
    table.add_row(['Schedule', f'{args.sche}'])
    table.add_row(['Init learning rate', f'{args.lr}'])
    table.add_row(['Train size', f'{args.n_train}'])
    table.add_row(['Validation size', f'{args.n_val}'])
    table.add_row(['Device [id]', {args.device.type + ':/' + args.gpus}])
    table.add_row(['Input channels', f'{args.n_channels}'])
    table.add_row(['Output channels', f'{args.n_classes}'])
    table.add_row(['Image sizes', f'{(args.img_size, args.img_size)}'])
    table.add_row(['Mean & Std', f'{(mean, std)}'])
    table.add_row(['Datatime', datetime.now().strftime('%y-%m-%d-%H-%M-%S')])
    table.add_row(['Paramters', "%.2fM" % (sum(x.numel() for x in args.net.parameters()) / 1e+6)])
    logx.msg(str(table) + '\n')

    train_net()

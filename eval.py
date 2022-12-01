import torch
from tqdm import tqdm
from metrics import *
import torch.nn.functional as F

def eval_net(criterion, logx, epoch, args, viz, isMemory):
    dc = 0.  # dice coefficient / f1-score
    se = 0.  # sensitivity score / recall rate
    sp = 0.  # specificity score
    jc = 0.  # jaccard score / iou score
    acc = 0.  # accuracy score
    auc = 0.  # area under curve score
    step, epoch_loss, epoch_seg_loss, epoch_class_loss = 0, 0., 0., 0.

    with torch.no_grad():
        with tqdm(total=args.n_val, desc='Validation round', unit='img', leave=False) as pbar:
            args.net.eval()

            for batch in args.val_loader:
                imgs, true_masks  = batch['image'], batch['mask']

                imgs = imgs.to(device=args.device, dtype=torch.float32)
                mask_type = torch.float32 if args.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=args.device, dtype=mask_type)

                if args.ifCrossImage:
                    mask_pred = args.net(imgs, true_masks, isMemory)  
                    loss , seg_loss, class_loss = criterion(mask_pred, true_masks, is_loss=True)
                    if isinstance(mask_pred[0], list):
                        mask_pred = mask_pred[0][0]
                    else:
                        mask_pred = mask_pred[0]  
                else:
                    if args.arch == 'DCRNet':
                        mask_pred = args.net(imgs, flag='test')
                        loss = criterion(mask_pred, true_masks)
                    elif args.arch in ["UNet_Res_cross"]:
                        mask_pred, A, bank_gt = args.net(imgs, true_masks)
                        loss = criterion(mask_pred, true_masks, A, bank_gt)
                    else:
                        mask_pred = args.net(imgs)
                        loss = criterion(mask_pred, true_masks)

                step += 1



                logx.add_scalar('Loss/val', loss.item(), epoch * args.n_val / imgs.shape[0] + step)
                epoch_loss += loss.item()
                if args.ifCrossImage or args.cls or args.arch.endswith('_scon'):
                    logx.add_scalar('Loss/val', seg_loss.item(), epoch * args.n_val / imgs.shape[0] + step)
                    epoch_seg_loss += seg_loss.item()
                    logx.add_scalar('Loss/val', class_loss.item(), epoch * args.n_val / imgs.shape[0] + step)
                    epoch_class_loss += class_loss.item()

                if isinstance(mask_pred, list):
                    mask_pred = mask_pred[0]
                else:
                    mask_pred = F.sigmoid(mask_pred)

                acc += get_accuracy(mask_pred, true_masks)
                dc += get_f1_score(mask_pred, true_masks)
                se += get_sensitivity(mask_pred, true_masks)
                sp += get_specificity(mask_pred, true_masks)
                jc += get_jaccard_score(mask_pred, true_masks)

                pbar.update(imgs.shape[0])

    num = args.n_val // imgs.shape[0]

    val_score = {
        'jc': jc / num,
        'dc': dc / num,
        'acc': acc / num,
        'sp': sp / num,
        'se': se / num,
        'auc': auc / num
    }
    mean_epoch_loss = epoch_loss / num

    if args.ifCrossImage or args.cls or args.arch.endswith('_scon'):
        mean_epoch_seg_loss = epoch_seg_loss / num
        mean_epoch_class_loss = epoch_class_loss / num
        return val_score, [mean_epoch_loss, mean_epoch_seg_loss, mean_epoch_class_loss]
    else:
        return val_score, mean_epoch_loss

import os
import cv2
import json
import numpy as np
import ttach as tta
from tqdm import tqdm
from runx.logx import logx
from sklearn import metrics
from albumentations import Resize
from prettytable import PrettyTable

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from opts import Opts
from dataloader import *
from metrics import *
from utils.preprocess import get_properties
import pandas as pd

import torch.nn.functional as F

def one_hot(masks, label):
    #masks:[ b, 1, h, w] label: [1, 0, 0] 正常  [0, 1, 0] 良性  [0, 0, 1] 恶性
    #                                   0 正常          1 良性         2  恶性
    shp_x = (masks.size(0), 3, masks.size(2), masks.size(3))
    cls = torch.argmax(label, dim=1) # 类别

    for i in range(masks.size(0)):
        temp = masks[i,...]
        temp[temp == 1] = cls[i]
        masks[i,...] = temp

    with torch.no_grad():
        masks = masks.long()
        y_onehot = torch.zeros(shp_x)
        if masks.device.type == "cuda":
            y_onehot = y_onehot.cuda(masks.device.index)
        y_onehot.scatter_(1, masks, 1)
    return y_onehot

def preprocess(image):
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)
    aug = Resize(args.img_size, args.img_size)
    augmented = aug(image=image)
    image = augmented['image']
    if image.max() > 1:
        image = (image - mean) / sd
    image = image.transpose((2, 0, 1))
    image = image.astype(np.float32)
    return image

def predict(pil_img, save_path, threshold=0.5):
    args.net.eval()
    img = torch.from_numpy(preprocess(pil_img))
    img = img.unsqueeze(0)
    img = img.to(device=args.device, dtype=torch.float32)

    with torch.no_grad():
        if args.ifCrossImage:
            output = args.net(img, img)
            if isinstance(output[0][0], list):
                output = output[0][0][0]
            else:
                output = output[0][0,...].unsqueeze(0)
        else:
            if args.arch == 'DCRNet':
                output = args.net(img, flag='test')
            else:
                if args.arch == 'DCRNet':
                    output = args.net(img, flag='test')
                else:
                    output = args.net(img)
        if isinstance(output, list):
            output = output[0]
        if args.ifSoftmax :
            output = F.softmax(output,dim=1)
        else:
            output = F.sigmoid(output)
        probs = output.squeeze(0)
        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor()
            ]
        )
        if args.ifSoftmax:
            for i in range(probs.size(0)):
                prob = tf(probs[i].cpu())
                full_mask = prob.squeeze().cpu().numpy()
                mask = full_mask > threshold
                mask = mask.astype(np.int) * 255
                mask = np.expand_dims(mask, axis=2)
                id = save_path.split('/')[-1]
                save_path_ = save_path.replace(id,id.split('.')[0]+'_'+str(i)+'.'+id.split('.')[-1])
                cv2.imwrite(save_path_, mask)
        else:
            probs = tf(probs.cpu())
            full_mask = probs.squeeze().cpu().numpy()
            mask = full_mask > threshold
            mask = mask.astype(np.int) * 255
            mask = np.expand_dims(mask, axis=2)
            cv2.imwrite(save_path, mask)


##['test']这里要注意改到跟dataloader的一样才可以
def get_id():
    if args.dataset in ["busi", "datasetB", "kvasir"]:
        file_name = 'dataset' + args.fold + '.json'
        return json.load(open(os.path.join('data', args.dataset, file_name), 'r'))['test']
    return json.load(open(os.path.join('data', args.dataset, 'dataset.json'), 'r'))['test']


def eval_all_dataset():
    args.net.eval()
    acc = 0.
    dc = 0.
    se = 0.
    sp = 0.
    jc = 0.
    auc = 0.
    dc_b, se_b, sp_b, jc_b, acc_b, auc_b = 0., 0., 0., 0, 0., 0.
    dc_m, se_m, sp_m, jc_m, acc_m, auc_m = 0., 0., 0., 0, 0., 0.
    ids = get_id()
    if args.ifCrossImage:
        to_csv = pd.DataFrame(columns=('file_id', 'Dice', 'Class'))
    else:
        to_csv = pd.DataFrame(columns=('file_id', 'Dice'))
    with torch.no_grad():
        with tqdm(total=n_test, desc='Test', unit='img', leave=False) as pbar:
            for i, batch in enumerate(test_loader):
                original_img = cv2.imread(os.path.join(args.dir_img, ids[i]))
                predict(original_img, save_path=os.path.join(args.dir_result, ids[i]))

                # Predict the image
                if args.ifCrossImage:
                    imgs, true_masks, label, imgs2, true_masks2, label2 = \
                        batch['image'], batch['mask'], batch['label'], batch['image'], batch['mask'], batch['label']
                    if args.same:
                        imgs3, true_masks3 = batch['image3'], batch['mask3']
                else:
                    imgs, true_masks, label = batch['image'], batch['mask'], batch['label']

                imgs = imgs.to(device=args.device, dtype=torch.float32)
                mask_type = torch.float32 if args.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=args.device, dtype=mask_type)
                label = label.to(device=args.device, dtype=torch.float32)
                if args.ifCrossImage:
                    imgs2 = imgs2.to(device=args.device, dtype=torch.float32)
                    true_masks2 = true_masks2.to(device=args.device, dtype=mask_type)
                    label = label.to(device=args.device, dtype=torch.float32)
                    label2 = label2.to(device=args.device, dtype=torch.float32)
                    if args.same:
                        imgs3 = imgs3.to(device=args.device, dtype=torch.float32)
                        true_masks3 = true_masks3.to(device=args.device, dtype=mask_type)

                #转one hot 编码
                if args.ifSoftmax:
                    true_masks = one_hot(true_masks, label)
                    if args.ifCrossImage:
                        true_masks2 = one_hot(true_masks2, label2)
                        if args.same:
                            true_masks3 = one_hot(true_masks3, label2)

                if args.ifCrossImage:
                    mask_pred = args.net(imgs, imgs)  # 返回的是列表 包含 列表
                    # class_id = torch.argmax(mask_pred[1], dim=1)[0]
                    if isinstance(mask_pred[0][0], list):
                        mask_pred = mask_pred[0][0][0]
                    else:
                        mask_pred = mask_pred[0][0,...].unsqueeze(0)  # 保持后面的一致，计算分割的精度
                else:
                    mask_pred = args.net(imgs)

                if isinstance(mask_pred, list):
                    mask_pred = mask_pred[0]

                if args.ifSoftmax :
                    mask_pred = F.softmax(mask_pred, dim=1)
                    # mask_pred = mask_pred[:, 1:3,...]
                    # true_masks = true_masks[:,1:3,...]
                else:
                    mask_pred = F.sigmoid(mask_pred)

                if args.ifSoftmax:
                    acc_b += get_accuracy(mask_pred[:,1,...].unsqueeze(1), true_masks[:,1,...].unsqueeze(1))
                    dc_b += get_f1_score(mask_pred[:,1,...].unsqueeze(1), true_masks[:,1,...].unsqueeze(1))
                    se_b += get_sensitivity(mask_pred[:,1,...].unsqueeze(1), true_masks[:,1,...].unsqueeze(1))
                    sp_b += get_specificity(mask_pred[:,1,...].unsqueeze(1), true_masks[:,1,...].unsqueeze(1))
                    jc_b += get_jaccard_score(mask_pred[:,1,...].unsqueeze(1), true_masks[:,1,...].unsqueeze(1))

                    acc_m += get_accuracy(mask_pred[:,2,...].unsqueeze(1), true_masks[:,2,...].unsqueeze(1))
                    dc_m += get_f1_score(mask_pred[:,2,...].unsqueeze(1), true_masks[:,2,...].unsqueeze(1))
                    se_m += get_sensitivity(mask_pred[:,2,...].unsqueeze(1), true_masks[:,2,...].unsqueeze(1))
                    sp_m += get_specificity(mask_pred[:,2,...].unsqueeze(1), true_masks[:,2,...].unsqueeze(1))
                    jc_m += get_jaccard_score(mask_pred[:,2,...].unsqueeze(1), true_masks[:,2,...].unsqueeze(1))

                    acc += (get_accuracy(mask_pred[:,1,...].unsqueeze(1), true_masks[:,1,...].unsqueeze(1)) +\
                            get_accuracy(mask_pred[:,2,...].unsqueeze(1), true_masks[:,2,...].unsqueeze(1)))/2.0
                    dc += (get_f1_score(mask_pred[:,1,...].unsqueeze(1), true_masks[:,1,...].unsqueeze(1)) + \
                           get_f1_score(mask_pred[:,2,...].unsqueeze(1), true_masks[:,2,...].unsqueeze(1)))/2.0
                    se += (get_sensitivity(mask_pred[:,1,...].unsqueeze(1), true_masks[:,1,...].unsqueeze(1)) + \
                           get_sensitivity(mask_pred[:,2,...].unsqueeze(1), true_masks[:,2,...].unsqueeze(1)))/2.0
                    sp += (get_specificity(mask_pred[:,1,...].unsqueeze(1), true_masks[:,1,...].unsqueeze(1)) +
                           get_specificity(mask_pred[:,2,...].unsqueeze(1), true_masks[:,2,...].unsqueeze(1)))/2.0
                    jc += (get_jaccard_score(mask_pred[:,1,...].unsqueeze(1), true_masks[:,1,...].unsqueeze(1)) + \
                           get_jaccard_score(mask_pred[:,2,...].unsqueeze(1), true_masks[:,2,...].unsqueeze(1)))/2.0

                    # 保存测试集各个文件dice
                    dc_tmp = ( get_f1_score(mask_pred[:,1,...].unsqueeze(1), true_masks[:,1,...].unsqueeze(1)) \
                               + get_f1_score(mask_pred[:,2,...].unsqueeze(1), true_masks[:,2,...].unsqueeze(1)) )/2.0
                    to_csv = to_csv.append([{'file_id': ids[i], 'Dice': dc_tmp}], ignore_index=True)
                else:
                    acc += get_accuracy(mask_pred, true_masks)
                    dc += get_f1_score(mask_pred, true_masks)
                    se += get_sensitivity(mask_pred, true_masks)
                    sp += get_specificity(mask_pred, true_masks)
                    jc += get_jaccard_score(mask_pred, true_masks)

                    # 保存测试集各个文件dice
                    dc_tmp = get_f1_score(mask_pred, true_masks)
                    to_csv = to_csv.append([{'file_id': ids[i], 'Dice': dc_tmp}], ignore_index=True)

                fpr, tpr, thresholds = metrics.roc_curve(
                    true_masks.flatten().cpu().detach().numpy().astype(np.uint8),
                    mask_pred.flatten().cpu().detach().numpy(), pos_label=1)
                auc += metrics.auc(fpr, tpr)
                pbar.update(imgs.shape[0])
    to_csv.to_csv(os.path.join(args.dir_result, "result.csv"))
    num = n_test // imgs.shape[0]
    test_scores = {'jc': jc / num, 'dc': dc / num, 'acc': acc / num,
                   'sp': sp / num, 'se': se / num, 'auc': auc / num}
    if args.ifSoftmax:
        test_scores_b = {'jc_b': jc_b / num, 'dc_b': dc_b / num, 'acc_b': acc_b / num,
                       'sp_b': sp_b / num, 'se_b': se_b / num, 'auc_b': auc_b / num}
        test_scores_m = {'jc_m': jc_m / num, 'dc_m': dc_m / num, 'acc_m': acc_m / num,
                       'sp_m': sp_m / num, 'se_m': se_m / num, 'auc_m': auc_m / num}
        test_scores = [test_scores, test_scores_b, test_scores_m]

    return test_scores


if __name__ == '__main__':
    args = Opts().init()
    args.net.load_state_dict(
        # torch.load(os.path.join(args.dir_ckpt, f'{args.dataset}_{args.arch}_{args.exp_id}.pth')))
        torch.load(os.path.join(args.dir_ckpt, f'{args.dataset}_{args.arch}_{args.exp_id}_best_val.pth')))
    # print(args.net)

    if not os.path.exists(args.dir_result):
        os.makedirs(args.dir_result)

    if args.tta:
        """ reference: https://github.com/qubvel/ttach """
        trans = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.Rotate90(angles=[0, 90, 180, 270]),
            ]
        )
        args.net = tta.SegmentationTTAWrapper(args.net, trans, merge_mode='mean')

    results = get_properties(args=args)
    mean, sd = results['mean'], results['sd']
    if args.dataset == 'busi':
        if args.ifCrossImage:
            test_dataset = busi_cross(args, mean, sd, flag='test')
        else:
            test_dataset = busi(args, mean, sd, flag='test')
        mean = [0.485, 0.456, 0.406]
        sd = [0.229, 0.224, 0.225]
    elif args.dataset == 'datasetB':
        test_dataset = busi_cross(args, mean, sd, flag='test')
        mean = [0.485, 0.456, 0.406]
        sd = [0.229, 0.224, 0.225]
    elif args.dataset == 'kvasir':
        test_dataset = busi_cross(args, mean, sd, flag='test')
        mean = [0.485, 0.456, 0.406]
        sd = [0.229, 0.224, 0.225]
    else:
        test_dataset = []
    n_test = len(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=0, pin_memory=False, drop_last=False)

    logx.initialize(logdir=args.dir_ckpt, coolname=True, tensorboard=True)
    logx.msg('Start testing...\n')
    table = PrettyTable(["key", "value"])
    table.align = 'l'
    table.add_row(['Dataset', f'{args.dataset}'])
    table.add_row(['Experiment id', f'{args.exp_id}'])
    table.add_row(['Test time augmentation', f'{args.tta}'])
    table.add_row(['Device [id]', {args.device.type + ':/' + args.gpus}])
    table.add_row(['Test size', f'{n_test}'])
    table.add_row(['Checkpoint dir', f'{args.dir_ckpt}'])
    logx.msg(str(table) + '\n')
    scores = eval_all_dataset()
    if args.ifSoftmax:
        for j in range(len(scores)):
            temp = scores[j]
            res = PrettyTable(temp.keys())
            res.align = 'l'
            res.add_row(temp.values())
            logx.msg(str(res) + '\n')
    else:
        res = PrettyTable(scores.keys())
        res.align = 'l'
        res.add_row(scores.values())
        logx.msg(str(res) + '\n')

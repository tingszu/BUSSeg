import os
import cv2
import time
import numpy as np
import ttach as tta
from tqdm import tqdm

import torch
from torchvision import transforms
from albumentations import Resize

from opts import Opts
from utils.preprocess import get_properties


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


def get_results(predictions):
    with tqdm(total=args.n_test, desc=f'predict', unit='img') as pbar:
        for index, i in enumerate(os.listdir(args.dir_test)):
            save_path = os.path.join(args.dir_result, i)
            pil_img = cv2.imread(os.path.join(args.dir_test, i))

            mask = predictions[0][index]
            for j in range(len(predictions[1:])):
                mask += predictions[j][index]
            mask = mask / kfold
            mask = mask > 0.5
            mask = mask.astype(np.int) * 255
            mask = np.expand_dims(mask, axis=2)
            aug = Resize(pil_img.shape[0], pil_img.shape[1])
            augmented = aug(image=pil_img, mask=mask)
            mask = augmented['mask']
            cv2.imwrite(save_path, mask)
            pbar.update(1)


def get_npy(model, idx):
    with tqdm(total=args.n_test, desc=f'start', unit='img') as pbar:
        kfolds = []
        for index, i in enumerate(os.listdir(args.dir_test)):
            pil_img = cv2.imread(os.path.join(args.dir_test, i))
            model.eval()
            img = torch.from_numpy(preprocess(pil_img))
            img = img.unsqueeze(0)
            img = img.to(device=args.device, dtype=torch.float32)

            with torch.no_grad():
                output = model(img)
                probs = output.squeeze(0)
                tf = transforms.Compose(
                    [
                        transforms.ToPILImage(),
                        transforms.ToTensor()
                    ]
                )
                probs = tf(probs.cpu())
                full_mask = probs.squeeze().cpu().numpy()
            kfolds.append(full_mask)
            pbar.update(1)
    print("Successful saved the k-fold{}.".format(idx))
    np.save(os.path.join(args.dir_ckpt, "kfold{}.npy".format(idx)), kfolds)
    print("-" * 100)


if __name__ == '__main__':
    args = Opts().init()

    kfold = 5
    masks = []
    results = get_properties(args=args)
    mean, sd = results['mean'], results['sd']

    for k in range(kfold):
        net = args.net
        net.load_state_dict(
            torch.load(os.path.join(args.dir_ckpt, 'MODEL{}.pth'.format(k))))
        if not os.path.exists(args.dir_result):
            os.makedirs(args.dir_result)
        if args.tta:
            """ tta: https://github.com/qubvel/ttach """
            net = tta.SegmentationTTAWrapper(args.net, tta.aliases.d4_transform(), merge_mode='mean')
        masks.append(get_npy(net, idx=k))

    time.sleep(20)

    for k in range(kfold):
        masks.append(np.load(os.path.join('logs', 'thyroid', 'EXP_CV_NET_UADNet', "kfold{}.npy".format(k))))
    for k in range(kfold):
        masks.append(np.load(os.path.join('logs', 'thyroid', 'EXP_CV_NET_UNet', "kfold{}.npy".format(k))))
    for k in range(kfold):
        masks.append(np.load(os.path.join('logs', 'thyroid', 'EXP_CVFU_NET_UADNet', "kfold{}.npy".format(k))))

    get_results(masks)

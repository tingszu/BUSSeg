import os
import cv2
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
    results = get_properties(args=args)
    mean, sd = results['mean'], results['sd']
    if image.max() > 1:
        image = (image - mean) / sd
    image = image.transpose((2, 0, 1))
    image = image.astype(np.float32)
    return image


def predict(full_img, threshold=0.5):
    args.net.eval()
    img = torch.from_numpy(preprocess(full_img))
    img = img.unsqueeze(0)
    img = img.to(device=args.device, dtype=torch.float32)

    with torch.no_grad():
        output = args.net(img)
        probs = output.squeeze(0)
        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor()
            ]
        )
        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()
    return full_mask > threshold


def submit():
    with tqdm(total=args.n_test, desc=f'predict', unit='img') as pbar:
        for index, i in enumerate(os.listdir(args.dir_test)):
            save_path = os.path.join(args.dir_result, i)
            pil_img = cv2.imread(os.path.join(args.dir_test, i))
            mask = predict(full_img=pil_img, threshold=0.5)

            # Threshold filter
            mask = mask * (mask.sum() > args.threshold).astype(np.int)

            aug = Resize(pil_img.shape[0], pil_img.shape[1])
            augmented = aug(image=pil_img, mask=mask)
            mask = augmented['mask']
            mask = mask.astype(np.int) * 255
            mask = np.expand_dims(mask, axis=2)
            cv2.imwrite(save_path, mask)
            pbar.update(1)


if __name__ == '__main__':
    args = Opts().init()
    args.net.load_state_dict(
        torch.load(os.path.join(args.dir_ckpt, f'{args.dataset}_{args.arch}_{args.exp_id}.pth')))
    if not os.path.exists(args.dir_result):
        os.makedirs(args.dir_result)
    if args.tta:
        """ reference: https://github.com/qubvel/ttach """
        args.net = tta.SegmentationTTAWrapper(args.net, tta.aliases.d4_transform(), merge_mode='mean')
    submit()

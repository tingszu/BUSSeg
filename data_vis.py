import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from albumentations import Resize

import torch
from torchvision import transforms

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


def plot_img_two(img, pred, name):
    fig, ax = plt.subplots(1, 2)
    ax[0].set_title('image')
    ax[0].imshow(img, cmap='gray')
    ax[0].axis('off')
    ax[1].set_title(f'gt')
    ax[1].imshow(pred, cmap='gray')
    ax[1].axis('off')
    ax[2].set_title(f'pred')
    ax[2].imshow(pred, cmap='gray')
    ax[2].axis('off')
    plt.axis('off')
    plt.savefig(os.path.join(args.dir_vis, name))


def visualize():
    ids = [file for file in os.listdir(args.dir_test)
           if not file.startswith('.')]

    with tqdm(total=len(ids), desc=f'visualize', unit='img') as pbar:
        for idx in ids:
            img_file = os.path.join(args.dir_test, idx)
            original_img = cv2.imread(img_file)
            pred_mask = predict(original_img)
            pred_mask = pred_mask.astype(np.int)
            aug = Resize(original_img.shape[0], original_img.shape[1])
            augmented = aug(image=original_img, mask=pred_mask)
            pred_mask = augmented['mask']
            plot_img_two(original_img, pred_mask, idx)
            pbar.update(1)


if __name__ == '__main__':
    args = Opts().init()
    args.net.load_state_dict(
        torch.load(os.path.join(args.dir_ckpt, f'{args.dataset}_{args.arch}_{args.exp_id}.pth')))

    if not os.path.exists(args.dir_vis):
        os.makedirs(args.dir_vis)

    visualize()

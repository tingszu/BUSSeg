from tqdm import tqdm
import os
import numpy as np
import cv2


def unify_mask():
    test_dir = '../results/thyroid/CVPFU/'
    total = len(list(os.listdir(test_dir)))
    with tqdm(total=total, desc=f'process_mask', unit='img') as pbar:
        for i in os.listdir(test_dir):
            path = f"./thyroid/CVPFU/{os.path.splitext(i)[0]}.PNG"
            mask = cv2.imread(test_dir + i, 0)
            mask[mask > 100] = 255
            mask[mask <= 100] = 0
            mask = mask.astype(np.int)
            mask = np.expand_dims(mask, axis=2)
            cv2.imwrite(path, mask)
            pbar.update(1)


unify_mask()


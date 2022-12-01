import os
import json
import numpy as np
from glob import glob
from collections import OrderedDict


np.random.seed(12345)


def split_dataset():
    json_file = OrderedDict()
    split_train_val = False
    n_flod = True
    flod_num = 5
    data_type = '*).png'  # Please set according to your specific task.
    # masks = glob(os.path.join('/data/hxt', 'isic18', 'mask', data_type))
    filedir = '/data/hxt/busi'
    masks = sorted(glob(os.path.join(filedir, 'image/benign*.png')))
    masks_1 = [os.path.split(p)[-1] for p in masks]
    np.random.shuffle(masks_1)

    masks = sorted(glob(os.path.join(filedir, 'image/malignant*.png')))
    masks_2 = [os.path.split(p)[-1] for p in masks]
    np.random.shuffle(masks_2)

    masks = sorted(glob(os.path.join(filedir, 'image/normal*.png')))
    masks_3 = [os.path.split(p)[-1] for p in masks]
    np.random.shuffle(masks_3)

    mask = [masks_1, masks_2, masks_3]

    for i in range(0,flod_num):
        p_train, p_val = 0.8, 0.2  #
        val = []
        train = []
        for j in range(len(mask)):
            n_val_start = int(i * (len(mask[j]) * p_val))
            n_val_end = int((i+1)*(len(mask[j]) * p_val))
            n_train_start_1 = 0
            n_train_end_1 = n_val_start
            n_train_start_2 = n_val_end
            n_train_end_2 = int(len(mask[j]))

            val.append(mask[j][n_val_start: n_val_end])
            train.append(mask[j][n_train_start_1:n_train_end_1] + mask[j][n_train_start_2:n_train_end_2])

        val = val[0] + val[1]+ val[2]
        np.random.shuffle(val)
        json_file['val'] = val
        # train 
        train = train[0] + train[1] + train[2]
        np.random.shuffle(train)
        json_file['train'] = train

        file_name = 'datasetN' + str(i) + '.json'
        if os.path.exists(os.path.join('', file_name)):
            os.remove(os.path.join('', file_name))
        with open(os.path.join('', file_name), 'w') as f:
            json.dump(json_file, f)
        f.close()


if __name__ == '__main__':
    split_dataset()

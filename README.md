# Cross-image Dependency Modelling for Breast Ultrasound Segmentation

This is the official repo for the paper "Cross-image Dependency Modelling for Breast Ultrasound Segmentation".
**Due to the confidentiality agreement in commercial cooperation, we only provide codes of core modules and the whole trainable models for the convenience of comparisons.**
We have released the codes of core modules and more content will be added later. This paper is accepted by TMI.

## 1. Datasets
* **BUSI**: A public BUS datasets from [Dataset of breast ultrasound images](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset)
* **UDIAT DatasetB**: A public BUS datasets from [Automated breast ultrasound lesions detection using convolutional neural networks]
* **Nerve Ultrasound Dataset**: A dataset from the Kaggle ultrasound nerve segmentation challenge(https://www.kaggle.com/c/ultrasound-nerve-segmentation)

## 2. Abstract
We present a novel deep network (namely BUSSeg) equipped with both within- and cross-image long-range dependency modeling for automated lesions segmentation from breast ultrasound images, which is a quite daunting task due to (1) the large variation of breast lesions, (2) the ambiguous lesion boundaries, and (3) the existence of speckle noise and artifacts in ultrasound images. Our work is motivated by the fact that most existing methods only focus on modeling the within-image dependencies while neglecting the cross-image dependencies, which are essential for this task under limited training data and noise. We first propose a novel cross-image dependency module (CDM) with a cross-image contextual modeling scheme and a cross-image dependency loss (CDL) to capture more consistent feature expression and alleviate noise interference. Compared with existing cross-image methods, the proposed CDM has two merits. First, we utilize more complete spatial features instead of commonly used discrete pixel vectors to capture the semantic dependencies between images, mitigating the negative effects of speckle noise and making the acquired features more representative. Second, the proposed CDM includes both intra- and inter-class contextual modeling rather than just extracting homogeneous contextual dependencies. Furthermore, we develop a parallel bi-encoder architecture (PBA) to tame a Transformer and a convolutional neural network to enhance BUSSeg's capability in capturing within-image long-range dependencies and hence offer richer features for CDM. We conducted extensive experiments on two representative public breast ultrasound datasets, and the results demonstrate that the proposed BUSSeg consistently outperforms state-of-the-art approaches in most metrics.

## 3. Requirements
```
CUDA/CUDNN
python=3.8.3
torch=1.8.0
torchvision
numpy
```
Download training model "busi_BUSSeg.pth" from:
https://drive.google.com/file/d/1ie-p0ZZdVUq05TzRIRuT40C1XMeQJttq/view?usp=sharing

## 4. Train/Test
Run the train script on busi dataset. The batch size we used is 4. If you do have enough GPU memory, the bacth size can be increased to 8 or 16 to save memory.
* Train
```
python train.py --arch BUSSeg --dataset busi -e 100 --fold 1 --exp_id 1 --img_size 384 --ifCrossImage
```
* Test
```
python test.py --arch BUSSeg  --dataset busi -e 100 --fold 1 --exp_id 1 --img_size 384 --ifCrossImage
```




<!--
## Acknowledgement
-->


## Citation
If you find BUSSeg useful, please cite our paper. This paper is accepted by TMI.

```
@article{wu2023cross,
  title={Cross-image Dependency Modelling for Breast Ultrasound Segmentation},
  author={Wu, Huisi and Huang, Xiaoting and Guo, Xinrong and Wen, Zhenkun and Qin, Jing},
  journal={IEEE Transactions on Medical Imaging},
  year={2023},
  publisher={IEEE}
}
```

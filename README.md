# BUSSeg

This is the official repo for the paper "Conjoining Within- and Cross-image Long-range Dependency Modeling for Breast Lesion Segmentation from Ultrasound Images".
We have released the codes of core modules and more content will be added later. This paper is under review (submitted to TMI).

# Datasets
* **BUSI**: A public BUS datasets from [Dataset of breast ultrasound images](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset)
* **UDIAT DatasetB**: A public BUS datasets from [Automated breast ultrasound lesions detection using convolutional neural networks]
* **Nerve Ultrasound Dataset**: A dataset from the Kaggle ultrasound nerve segmentation challenge(https://www.kaggle.com/c/ultrasound-nerve-segmentation)

# Abstract
We present a novel deep network (namely BUSSeg) equipped with both within- and cross-image long-range dependency modeling for automated lesions segmentation from breast ultrasound images, which is a quite daunting task due to (1) the large variation of breast lesions, (2) the ambiguous lesion boundaries, and (3) the existence
of massive speckle noise and artifacts in ultrasound images. Our work is motivated by the fact that most existing methods only focus on modeling the within-image dependencies while neglecting the cross-image dependencies, which are essential for this task under limited training data and massive noise. We first propose a novel crossimage dependency module (CDM) with a cross-image contextual modeling scheme and a cross-image dependency loss (CDL) to capture more consistent feature expression
and alleviate noise interference. Compared with existing cross-image methods, the proposed CDM has two merits. First, we utilize more complete spatial features instead
of commonly used discrete pixel vectors to capture the semantic dependencies between images, mitigating the negative effects of speckle noise and making the acquired
features more representative. Second, the proposed CDM includes both intra- and inter-class contextual modeling rather than just extracting homogeneous contextual dependencies. Furthermore, we develop a parallel bi-encoder architecture (PBA) to tame a Transformer and a convolutional neural network to enhance BUSSeg’s capability
in capturing within-image long-range dependencies and hence offer richer features for CDM. We conducted extensive experiments on two representative public breast
ultrasound datasets, and the results demonstrate that the proposed BUSSeg consistently outperforms state-of-the-art approaches in most metrics.

# Requirements
```
CUDA/CUDNN
python=3.8.3
torch=1.8.0
torchvision
numpy
```

<!--
# Acknowledgement
-->


# Citation
If you find BUSSeg useful, please cite our paper. This paper is under review (submitted to TMI).

```
@article{BUSSeg,
    title = {Conjoining Within- and Cross-image Long-range Dependency Modeling for Breast Lesion Segmentation from Ultrasound Images},
    journal = {IEEE Transactions on Medical Imaging},
    volume = {XX},
    pages = {XXXXXX},
    year = {2022},
    issn = {XXXXXXXXXXX},
    doi = {XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX},
    url = {XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX},
    author = {Huisi Wu and Xiaoting Huang and Xinrong Guo and Zhenkun Wen and Jing Qin},
}
```

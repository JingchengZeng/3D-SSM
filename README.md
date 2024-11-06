# 3D-SSM: A novel 3D selective scan module for remote sensing change detection

![Framework](https://cdn.jsdelivr.net/gh/LonelyProceduralApe/BlogImage@main/2024/11_8117ee91f006d8850a58bb8ff608b868.png)

[![GitHub stars](https://badgen.net/github/stars/zmoka-zht/CDMamba)](https://github.com/zmoka-zht/CDMamba)


## Introduction

This repository is the code implementation of the paper 

The current branch has been tested on Linux system, PyTorch 2.1.1 and CUDA 11.8, supports Python 3.10.


## Updates

🌟 **2024.11.06** Released the 3D-SSM project.

## Installation

### Requirements

- Linux system, Windows is not tested, depending on whether `causal-conv1d` and `mamba-ssm` can be installed
- Python 3.8+, recommended 3.10
- PyTorch 2.0 or higher, recommended 2.1.1
- CUDA 11.8 or higher, recommended 11.8

### Environment Installation

It is recommended to use VMamba for installation. The following commands will create a virtual environment named `3d-ssm` and install PyTorch. In the following installation steps, the default installed CUDA version is **11.8**. 

Note: If you are experienced with PyTorch and have already installed it, you can skip to the next section. Otherwise, you can follow the steps below.

**Step 1**: Create a virtual environment named `3d-ssm` and activate it.

```shell
conda create -n 3d-ssm python=3.10
conda activate 3d-ssm
```

**Step 2**: Install dependencies.

```shell
pip install -r requirements.txt
```
**Note**: If importing mamba fails, please download the corresponding package at https://github.com/state-spaces/mamba/releases.

## Dataset Preparation


### Remote Sensing Change Detection Dataset

We provide the method of preparing the remote sensing change detection dataset used in the paper.

- [WHU-CD Dataset](Fully convolutional networks for multisource building extraction from an open aerial and satellite imagery data set)

- [LEVIR-CD Dataset](A spatial-temporal attention-based method and a new dataset for remote sensing image change detection.)

- [CDD Dataset](change detection in remote sensing images using conditional adversarial networks)
- [SYSU Dataset](A deeply supervised attention metric-based network and an open aerial image dataset for remote sensing change detection.)
- [DSIFN Dataset](A deeply supervised image fusion network for change detection in high resolution bi-temporal remote sensing images.)


#### Organization Method

You can also choose other sources to download the data, but you need to organize the dataset in the following format：

```
${DATASET_ROOT} # Dataset root directory, for example: /home/username/data/LEVIR-CD
├── train
│   ├── A
│   │   ├── 00001.png
│   │	├── ...
│   ├── B
│   │   ├── 00001.png
│   │	├── ...
│   └── label
│   	├── 00001.png
│   	└── ...
├── val
│   ├── A
│   │   ├── 00001.png
│   │	├── ...
│   ├── B
│   │   ├── 00001.png
│   │	├── ...
│   └── label
│   	├── 00001.png
│   	└── ...
├── test
│   ├── A
│   │   ├── 00001.png
│   │	├── ...
│   ├── B
│   │   ├── 00001.png
│   │	├── ...
│   └── label
│   	├── 00001.png
│   	└── ...
└── list
    ├── train.txt
    ├── val.txt
    └── test.txt
```

## Model Training and Testing

All configuration for model training and testing are stored in the local folder `configs`

#### Example of Training on WHU-CD Dataset

```shell
python train.py --config configs/3dssm+whu+train.json --batch_size 16
```

#### Example of Testing on WHU-CD Dataset

```shell
python test.py --config configs/3dssm+whu+train.json
```
## Citation

This project is based on the Haotian Zhang, Keyan Chen, Chenyang Liu, Hao Chen, Zhengxia Zou, and Zhenwei Shi. Cdmamba: Remote sensing image change detection with mamba. arXiv preprint arXiv:2406.04207, 2024. 1, 3, 5.[](https://arxiv.org/abs/2406.04207)

This project is licensed under the [Apache 2.0 License](LICENSE).

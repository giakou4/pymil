# PyMIL

A PyTorch implementation of Self-Supervised Learning (SSL) methods

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/giakou4/pymil/LICENSE)
![size](https://img.shields.io/github/languages/code-size/giakou4/pymil)

## 1. Prerequisites

Before proceeding, create a [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) environment:

```shell
conda create -n pymil
```
   
Activate the newly created environment:

```shell
conda activate pymil
```

Once the environment is activated, install the required packages from the "requirements.txt" file using the following command:

```shell
pip install -r requirements.txt
```

## 2. Methods

The MIL method implementation was based on [AMLab](https://github.com/AMLab-Amsterdam/AttentionDeepMIL) implementation of [Attention-based Deep Multiple Instance Learning](https://arxiv.org/abs/1802.04712).

For the following section, assume a backbone, e.g., a ResNet-50, and an input image of size 28 (we converted the grayscale images of MNIST to RGB for demonstration) in a bag of random size (e.g., 10) of batch size 1, i.e.,

```python
import torchvision

backbone = torchvision.models.resnet50(pretrained=False)
feature_size = backbone.fc.in_features
backbone.fc = torch.nn.Identity()

x = torch.rand(1, 10, 3, 28, 28) # batch_size x bag_size x channels x height x width
```

### 2.1 Attention

TBD

### 2.2 Gated Attention

TBD

### 2.3 Mean-operation

TBD

### 2.4 Max-operation

TBD

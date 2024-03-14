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

The Attention method leverages an attention mechanism to weight instances within a bag, emphasizing those more relevant to the task. It dynamically adjusts focus over different parts of the data, enabling the model to learn which instances contribute most significantly to the bag's label.

```python
model = Attention(backbone, feature_size, L=500, D=128, K=1)
```

### 2.2 Gated Attention

Gated Attention extends the Attention approach by introducing a gating mechanism, which acts as an additional filter for the attention weights. This mechanism allows the model to not only focus on relevant instances but also control the flow of information, improving the model's ability to capture complex dependencies within the data.


```python
model = GatedAttention(backbone, feature_size, L=500, D=128, K=1)
```

### 2.3 Mean-operation

The Mean-operation in MIL aggregates instance representations within a bag by computing their mean. This operation provides a straightforward way to summarize the information across all instances, assuming equal importance, and produces a single representation for the entire bag. Mean-operation is non-trainable.

```python
model = Mean(backbone, feature_size, L=500)
```

### 2.4 Max-operation

Max-operation, similar to the Mean-operation, aggregates instance representations within a bag, but it does so by selecting the maximum value across instances for each feature. This approach allows the model to capture the most salient features present in any instance within the bag, emphasizing the instances with the highest impact on the bag's label. Max-operation is non-trainable.

```python
model = Max(backbone, feature_size, L=500)
```

## 3. Training

The models directly output the prediction of the bag, i.e., ```y_hat = model(x)``` and the loss can be calculated using the BCE loss as the criterion (see main.py).

```python
import torchvision
import torch
from torchvision import datasets, transforms
from loader import BagDataset
from builders import Attention

# get device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# initialize backbone (resnet50)
backbone = torchvision.models.resnet50(pretrained=False)
feature_size = backbone.fc.in_features
backbone.fc = torch.nn.Identity()

# initialize mil method
model = Attention(backbone, feature_size)
model = model.to(device)

# set transform
transform = transforms.Compose([
                                transforms.Grayscale(num_output_channels=3), # so we can use ResNet-50
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

# load MNIST and transform to bags
dataset = datasets.MNIST('./data', download=True, train=True, transform=transform)

# convert to bag dataset
dataset_bag = BagDataset(dataset=dataset, 
                         target_number=9, 
                         mean_bag_length=10, 
                         var_bag_length=2, 
                         num_bags=100, 
                         seed=1,
                         )
    
# set loaders
loader = torch.utils.data.DataLoader(dataset_bag, batch_size=1, shuffle=True)

# set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-3)

# set criterion
criterion = torch.nn.BCELoss()

# switch to train mode
model.train()

# epoch training
for epoch in range(10):
    for i, (x, y) in enumerate(loader):
        y = y[0]
        x, y = x.to(device), y.to(device)
        y_prob = model(x)

        # zero the parameter gradients
        model.zero_grad()

        # compute loss
        loss = criterion(y_prob, y.unsqueeze(1))
        
        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
```
## 5. Citation

In Bibtex format:

```bibtex
@misc{pymil2024giakoumoglou,
   author = {Nikolaos Giakoumoglou},
   title = {PyMIL: A PyTorch implementation of Multiple Instance Learning (MIL) methods},
   year = {2024},
   publisher = {GitHub},
   journal = {GitHub repository},
   howpublished = {\url{https://github.com/giakou4/pymil}},
}
```

## 5. Support
Reach out to me:
- [giakou4's email](mailto:giakou4@gmail.com "giakou4@gmail.com")


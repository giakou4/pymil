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
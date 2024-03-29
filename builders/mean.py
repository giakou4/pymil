import torch
from torch import nn


class Mean(nn.Module):
    """
    Mean operation in Multiple Instance Learning
    """
    def __init__(self, backbone, feature_size, L=500):
        super().__init__()
        self.L = L
        self.feature_size = feature_size

        self.backbone = backbone
        self.projector = nn.Sequential(
            nn.Linear(feature_size, self.L),
            nn.ReLU(),
            )
        self.classifier = nn.Sequential(
            nn.Linear(self.L, 1),
            nn.Sigmoid(),
            )

    def forward(self, x):
        x = x.squeeze(0)
        
        # prepNN
        h = self.backbone(x)
        h = h.view(-1, self.feature_size)
        h = self.fc(h)  # [b x L]
        
        # aggregate function
        z = torch.mean(h, 0)

        # afterNN
        y = self.classifier(z)
        return y
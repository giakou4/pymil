import torch
from torch import nn
import torch.nn.functional as F 


class Attention(nn.Module):
    """
    Attention-based Deep Multiple Instance Learning
    Link: https://arxiv.org/abs/1802.04712
    Implementation: https://github.com/AMLab-Amsterdam/AttentionDeepMIL
    """
    def __init__(self, backbone, feature_size, L=500, D=128, K=1):
        super().__init__()
        self.L = L
        self.D = D
        self.K = K # attention branches
        self.feature_size = feature_size

        self.backbone = backbone
        self.projector = nn.Sequential(
            nn.Linear(feature_size, self.L),
            nn.ReLU(),
            )
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D), # matrix V
            nn.Tanh(),
            nn.Linear(self.D, self.K), # matrix W
            )
        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid(),
            )

    def forward(self, x):
        x = x.squeeze(0)

        h = self.backbone(x)
        h = h.view(-1, self.feature_size)
        h = self.projector(h)  # [b x L]

        a = self.attention(h)  # [b x K]
        a = torch.transpose(a, 1, 0)  # [K x b]
        a = F.softmax(a, dim=1)  # softmax over b
            
        z = torch.mm(a, h)  # [K x L]

        y = self.classifier(z)
        return y

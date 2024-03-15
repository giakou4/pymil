import torch
from torch import nn
import torch.nn.functional as F 


class GatedAttention(nn.Module):
    """
    Gated Attention-based Deep Multiple Instance Learning
    Link: https://arxiv.org/abs/1802.04712
    Implementation: https://github.com/AMLab-Amsterdam/AttentionDeepMIL
    """
    def __init__(self, backbone, feature_size, M=500, L=128, ATTENTION_BRANCHES=1):
        super().__init__()
        self.M = 500
        self.L = 128
        self.ATTENTION_BRANCHES = 1
        self.feature_size = feature_size

        self.backbone = backbone
        self.projector = nn.Sequential(
            nn.Linear(feature_size, self.M),
            nn.ReLU(),
            )
        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh(),
            )
        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix U
            nn.Sigmoid(),
            )
        self.attention_weights = nn.Linear(self.L, self.ATTENTION_BRANCHES) # martix W
        self.classifier = nn.Sequential(
            nn.Linear(self.M*self.ATTENTION_BRANCHES, 1),
            nn.Sigmoid(),
            )

    def forward(self, x):
        x = x.squeeze(0)

        h = self.backbone(x)
        h = h.view(-1, self.feature_size)
        h = self.projector(h)  # [b x M]

        a_V = self.attention_V(h)  # [b x L]
        a_U = self.attention_U(h)  # [b x L]
        a = self.attention_weights(a_V * a_U) # element wise multiplication -> [b x ATTENTION_BRANCHES]
        a = torch.transpose(a, 1, 0)  # [ATTENTION_BRANCHES x b]
        a = F.softmax(a, dim=1)  # softmax over b

        z = torch.mm(a, h)  # [ATTENTION_BRANCHES x M]

        y = self.classifier(z)
        return y

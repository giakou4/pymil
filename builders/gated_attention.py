import torch
from torch import nn
import torch.nn.functional as F 


class GatedAttention(nn.Module):
    """
    Gated Attention-based Deep Multiple Instance Learning
    Link: https://arxiv.org/abs/1802.04712
    Implementation: https://github.com/AMLab-Amsterdam/AttentionDeepMIL
    """
    def __init__(self, backbone, feature_size, L=500, D=128, K=1):
        super().__init__()
        self.L = L
        self.D = D
        self.K = K
        self.feature_size = feature_size

        self.backbone = backbone
        self.projector = nn.Sequential(
            nn.Linear(feature_size, self.L),
            nn.ReLU(),
            )
        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D), # matrix V
            nn.Tanh(),
            )
        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D), # matrix U
            nn.Sigmoid(),
            )
        self.attention_weights = nn.Linear(self.D, self.K) # martix W
        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid(),
            )

    def forward(self, x):
        x = x.squeeze(0)

        h = self.backbone(x)
        h = h.view(-1, self.feature_size)
        h = self.projector(h)  # [b x L]

        a_V = self.attention_V(h)  # [b x D]
        a_U = self.attention_U(h)  # [b x D]
        a = self.attention_weights(a_V * a_U) # element wise multiplication -> [b x K]
        a = torch.transpose(a, 1, 0)  # [K x b]
        a = F.softmax(a, dim=1)  # softmax over b

        z = torch.mm(a, h)  # [K x L]

        y = self.classifier(z)
        return y
"""
https://arxiv.org/abs/2006.02797

"""

# import pytorch
import torch
from torch import Tensor
from torch import nn

# import activation functions
import activations.terelu.functional as F


class TEReLU(nn.Module):
    def __init__(self):
        super().__init__()

        self.alpha = 1.0
        self.beta = nn.Parameter(torch.Tensor(1).fill_(1.0))
        self.mu = 1.0

    def forward(self, x: Tensor):
        return F.terelu(x, alpha=self.alpha, beta=self.beta, mu=self.mu)

import torch

from torch import nn
from torch.nn import functional as F


class EvenPad2d(nn.Module):
    def __init__(self, kernel_size, padding_mode='reflect', value: float = 0.0):
        super().__init__()

        self.large = kernel_size - 1
        self.small = 0

        self.padding_mode = padding_mode
        self.value = value

    def forward(self, x):
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)

        x1 = F.pad(x1, [self.small, self.large, self.small, self.large], self.padding_mode, self.value)
        x2 = F.pad(x2, [self.large, self.small, self.small, self.large], self.padding_mode, self.value)
        x3 = F.pad(x3, [self.small, self.large, self.large, self.small], self.padding_mode, self.value)
        x4 = F.pad(x4, [self.large, self.small, self.large, self.small], self.padding_mode, self.value)

        return torch.cat([x1, x2, x3, x4], dim=1)

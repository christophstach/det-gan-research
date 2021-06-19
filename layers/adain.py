import torch
import torch.nn.functional as F
from torch import nn, Tensor


class AdaptiveInstanceNormalization2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.scale = nn.Conv2d(in_channels, out_channels, (1, 1))
        self.bias = nn.Conv2d(in_channels, out_channels, (1, 1))

        self.bn_scale = nn.BatchNorm2d(out_channels)
        self.bn_bias = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor, y: Tensor):
        scale = self.scale(y)
        bias = self.scale(y)

        scale = self.bn_scale(scale)
        bias = self.bn_bias(bias)

        scale = F.leaky_relu(scale, 0.2, inplace=True)
        bias = F.leaky_relu(bias, 0.2, inplace=True)

        x_std, x_mean = torch.std_mean(x, dim=[2, 3], keepdim=True)
        norm = scale * ((x - x_mean) / x_std) + bias

        return norm

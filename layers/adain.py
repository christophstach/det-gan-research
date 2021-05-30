import torch
from torch import nn, Tensor


class AdaIn2d(nn.Module):
    # https://github.com/CellEight/Pytorch-Adaptive-Instance-Normalization/blob/c7b3849b6715ff5568658288a960a6cf0ee82c7d/AdaIN.py#L4

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.scale = nn.Conv2d(in_channels, out_channels, (1, 1), (1, 1), (0, 0), bias=False)
        self.bias = nn.Conv2d(in_channels, out_channels, (1, 1), (1, 1), (0, 0), bias=False)

    def mu(self, x):
        """ Takes a (n,c,h,w) tensor as input and returns the average across
        it's spatial dimensions as (h,w) tensor [See eq. 5 of paper]"""
        return torch.sum(x, (2, 3)) / (x.shape[2] * x.shape[3])

    def forward(self, x: Tensor, y: Tensor):
        """ Takes a content embedding x and a style embedding y and changes
        transforms the mean and standard deviation of the content embedding to
        that of the style. [See eq. 8 of paper] Note the permutations are
        required for broadcasting"""
        scale = self.scale(y)
        bias = self.scale(y)

        return (
                scale * ((x.permute([2, 3, 0, 1]) - self.mu(x)) / self.sigma(x)) + bias
        ).permute([2, 3, 0, 1])


class AdaptiveInstanceNormalization2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.scale = nn.Conv2d(in_channels, out_channels, (1, 1), (1, 1), (0, 0), bias=False)
        self.bias = nn.Conv2d(in_channels, out_channels, (1, 1), (1, 1), (0, 0), bias=False)

    def forward(self, x: Tensor, y: Tensor):
        scale = self.scale(y)
        bias = self.scale(y)

        x_std, x_mean = torch.std_mean(x, dim=[2, 3], keepdim=True)
        norm = scale * ((x - x_mean) / x_std) + bias

        return norm

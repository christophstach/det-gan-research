import torch
import torch.nn as nn


# https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L120
class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()

        self.epsilon = epsilon

    def forward(self, x):
        x = x * torch.rsqrt(
            torch.mean(
                x.square(),
                dim=1,
                keepdim=True
            ) + self.epsilon
        )

        return x

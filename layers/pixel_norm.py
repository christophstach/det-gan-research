import torch.nn as nn


# https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L120
class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, alpha=1e-8):
        y = x.pow(2.).mean(dim=1, keepdim=True).add(alpha).sqrt()  # [N1HW]
        y = x / y  # normalize the input x volume
        return y
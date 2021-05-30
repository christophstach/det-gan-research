import torch
from kornia import filter2D
from torch import nn
from torch.nn.functional import conv2d

from layers.pad import EvenPad2d


class AvgPoolBlur(nn.Module):
    def __init__(self):
        super().__init__()

        self.pad = nn.ReflectionPad2d((1, 0, 1, 0))
        self.pool = nn.AvgPool2d((2, 2), (1, 1))

    def forward(self, x):
        x = self.pad(x)
        return self.pool(x)


class OnesBlur(nn.Module):
    def __init__(self):
        super().__init__()

        self.kernel = torch.ones(1, 3, 3)

    def forward(self, x):
        return filter2D(x, self.kernel)


class StyleBlur(nn.Module):
    def __init__(self):
        super().__init__()

        self.kernel = self.make_kernel([1, 3, 3, 1]).unsqueeze(0).unsqueeze(0)
        self.pad = EvenPad2d(4)

    @staticmethod
    def make_kernel(k):
        kernel = torch.tensor(k, dtype=torch.float32)

        if kernel.ndim == 1:
            kernel = kernel[None, :] * kernel[:, None]

        kernel /= kernel.sum()

        return kernel

    def forward(self, x):
        x = self.pad(x)
        kernel = self.kernel.to(x).repeat(x.shape[1], x.shape[1], 1, 1)
        return conv2d(x, kernel, stride=1)

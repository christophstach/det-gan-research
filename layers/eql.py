from typing import Tuple

import torch.nn.functional as F
from numpy import sqrt, prod
from torch import nn, empty


class EqlLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            nn.init.normal_(
                empty(out_features, in_features)
            )
        )

        if bias:
            self.bias = nn.Parameter(
                nn.init.uniform_(
                    empty(out_features)
                )
            )
        else:
            self.register_parameter('bias', None)

        self.scale = sqrt(2 / sqrt(in_features))

    def forward(self, x):
        return F.linear(x, self.weight * self.scale, self.bias)


class EqlConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int],
                 stride: Tuple[int, int] = (1, 1),
                 padding: Tuple[int, int, int, int] = (0, 0, 0, 0),
                 dilation: Tuple[int, int] = (1, 1),
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'reflect'):

        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        # define the weight and bias if to be used
        self.weight = nn.Parameter(
            nn.init.normal_(
                empty(out_channels, in_channels, *self.kernel_size)
            )
        )

        if bias:
            self.bias = nn.Parameter(
                nn.init.uniform_(
                    empty(out_channels)
                )
            )
        else:
            self.register_parameter('bias', None)

        fan_in = prod(self.kernel_size) * in_channels  # value of fan_in
        self.scale = sqrt(2 / sqrt(fan_in))

    def forward(self, x):
        return F.conv2d(
            F.pad(x, self.padding, mode=self.padding_mode),
            self.weight * self.scale,
            self.bias,
            self.stride,
            (0, 0),
            self.dilation,
            self.groups
        )

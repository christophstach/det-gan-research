import math
from typing import List

import torch
from torch import nn, Tensor
from torch.nn.utils import spectral_norm as sn

from layers import MinibatchStdDev
from layers.conv import EqlConv2d
from utils import create_downscale, create_activation_fn


class MsgDiscriminator(nn.Module):
    def __init__(self, d_depth, image_size, image_channels, score_dim, pack=1):
        super().__init__()

        downscale = 'avgpool'
        activation_fn = 'lrelu'
        eql = True

        class Conv(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                if eql:
                    self.conv = EqlConv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1), padding_mode='reflect')
                else:
                    self.conv = nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1), padding_mode='reflect')

            def forward(self, x):
                return self.conv(x)

        class FirstBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.compute1 = nn.Sequential(
                    Conv(in_channels, in_channels),
                    create_activation_fn(activation_fn, out_channels),
                )

                self.compute2 = nn.Sequential(
                    Conv(in_channels, out_channels),
                    create_activation_fn(activation_fn, out_channels),
                    create_downscale(downscale)
                )

            def forward(self, rgb):
                x = self.compute1(rgb)
                x = self.compute2(x)

                return x

        class IntermediateBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                if eql:
                    self.fromRGB = EqlConv2d(image_channels * pack, in_channels, (1, 1), (1, 1), (0, 0))
                else:
                    self.fromRGB = nn.Conv2d(image_channels * pack, in_channels, (1, 1), (1, 1), (0, 0))

                self.compute1 = nn.Sequential(
                    Conv(in_channels * 2, in_channels),
                    create_activation_fn(activation_fn, in_channels),
                )

                self.compute2 = nn.Sequential(
                    Conv(in_channels, out_channels),
                    create_activation_fn(activation_fn, out_channels),
                    create_downscale(downscale)
                )

            def forward(self, x, rgb):
                x = self.compute1(torch.cat([x, self.fromRGB(rgb)], dim=1))
                x = self.compute2(x)

                return x

        class LastBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                if eql:
                    self.fromRGB = EqlConv2d(image_channels * pack, in_channels, (1, 1), (1, 1), (0, 0))
                else:
                    self.fromRGB = nn.Conv2d(image_channels * pack, in_channels, (1, 1), (1, 1), (0, 0))

                self.compute1 = nn.Sequential(
                    MinibatchStdDev(),
                    Conv(in_channels * 2 + 1, in_channels),
                    create_activation_fn(activation_fn, in_channels),
                )

                self.compute2 = nn.Sequential(
                    sn(nn.Conv2d(in_channels, in_channels, (4, 4), (1, 1), (0, 0))),
                    create_activation_fn(activation_fn, out_channels),
                )

                if eql:
                    self.scorer = EqlConv2d(in_channels, out_channels, (1, 1), (1, 1), (0, 0))
                else:
                    self.scorer = nn.Conv2d(in_channels, out_channels, (1, 1), (1, 1), (0, 0))

            def forward(self, x, rgb):
                x = self.compute1(torch.cat([x, self.fromRGB(rgb)], dim=1))
                x = self.compute2(x)
                x = self.scorer(x)

                return x

        # END block declaration section

        channels = [
            image_channels,
            *[
                2 ** i * d_depth
                for i in range(2, int(math.log2(image_size)))
            ],
            score_dim
        ]

        self.blocks = nn.ModuleList()
        self.pack = pack

        for i, channel in enumerate(channels):
            if i == 0:  # first
                self.blocks.append(
                    FirstBlock(channel * self.pack, channels[i + 1])
                )
            elif 0 < i < len(channels) - 2:  # intermediate
                self.blocks.append(
                    IntermediateBlock(channel, channels[i + 1])
                )
            elif i < len(channels) - 1:  # last
                self.blocks.append(
                    LastBlock(channel, channels[i + 1])
                )

    def forward(self, rgbs: List[Tensor]):
        if self.pack > 1:
            rgbs = [
                torch.reshape(rgb, (-1, rgb.shape[1] * self.pack, rgb.shape[2], rgb.shape[3]))
                for rgb in rgbs
            ]

        x = rgbs[0]
        for i, (b, rgb) in enumerate(zip(self.blocks, rgbs)):
            if i == 0:
                x = b(x)
            else:
                x = b(x, rgb)

        return x

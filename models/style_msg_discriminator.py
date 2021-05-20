from typing import List

import math
import torch
from torch import nn, Tensor
from torch.nn.utils import spectral_norm as sn

from layers import MinibatchStdDev
from layers.eql import EqlConv2d, EqlLinear
from layers.reshape import Reshape
from utils import create_downscale, create_activation_fn


class StyleMsgDiscriminator(nn.Module):
    def __init__(self, d_depth, image_size, image_channels, score_dim, pack=1):
        super().__init__()

        downscale = 'bilinear'
        activation_fn = 'lrelu'
        eql = False

        class Conv(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                if eql:
                    self.conv = sn(
                        EqlConv2d(
                            in_channels,
                            out_channels,
                            (3, 3),
                            (1, 1),
                            (1, 1),
                            padding_mode='reflect'
                        )
                    )
                else:
                    self.conv = sn(
                        nn.Conv2d(
                            in_channels,
                            out_channels,
                            (3, 3),
                            (1, 1),
                            (1, 1),
                            padding_mode='reflect'
                        )
                    )

            def forward(self, x):
                return self.conv(x)

        class FirstBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.compute1 = nn.Sequential(
                    Conv(in_channels, in_channels),
                    create_activation_fn(activation_fn, in_channels)
                )

                self.compute2 = nn.Sequential(
                    Conv(in_channels, out_channels),
                    create_downscale(downscale, in_channels, out_channels),
                    create_activation_fn(activation_fn, out_channels)
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
                    create_activation_fn(activation_fn, in_channels)
                )

                self.compute2 = nn.Sequential(
                    Conv(in_channels, out_channels),
                    create_downscale(downscale, in_channels, out_channels),
                    create_activation_fn(activation_fn, out_channels)
                )

            def forward(self, x, rgb=None):
                x = torch.cat([x, self.fromRGB(rgb)], dim=1)
                x = self.compute1(x)
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
                    create_activation_fn(activation_fn, in_channels)
                )

                if eql:
                    self.compute2 = nn.Sequential(
                        sn(EqlConv2d(in_channels, in_channels, (4, 4), (1, 1), (0, 0))),
                        create_activation_fn(activation_fn, out_channels)
                    )
                else:
                    self.compute2 = nn.Sequential(
                        sn(nn.Conv2d(in_channels, in_channels, (4, 4), (1, 1), (0, 0))),
                        create_activation_fn(activation_fn, out_channels)
                    )

                if eql:
                    self.reparam = nn.Sequential(
                        Reshape(shape=(-1, in_channels)),
                        sn(EqlLinear(in_channels, out_channels * 2))
                    )
                else:
                    self.reparam = nn.Sequential(
                        Reshape(shape=(-1, in_channels)),
                        sn(nn.Linear(in_channels, out_channels * 2)),
                    )

            def forward(self, x, rgb=None):
                x = torch.cat([x, self.fromRGB(rgb)], dim=1)
                x = self.compute1(x)
                x = self.compute2(x)

                statistics = self.reparam(x)
                mu, log_variance = statistics.chunk(2, 1)
                std = log_variance.mul(0.5).exp_()

                epsilon = torch.randn(x.shape[0], score_dim).to(statistics)
                y = epsilon.mul(std).add_(mu)

                return y

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

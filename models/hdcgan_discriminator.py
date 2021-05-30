import math

import torch
from torch import nn, Tensor
from torch.nn.utils import spectral_norm as sn

from layers.pad import EvenPad2d
from layers.reshape import Reshape
from utils import create_activation_fn


class ComputeBlock(nn.Module):
    def __init__(self, channels, kernel_size: int, activation_fn: str):
        super().__init__()

        if kernel_size % 2 == 0:
            self.conv = nn.Sequential(
                EvenPad2d(kernel_size, 'reflect'),
                sn(nn.Conv2d(channels, channels, (kernel_size, kernel_size), (1, 1), (0, 0)))
            )
        else:
            padding = kernel_size // 2

            self.conv = sn(nn.Conv2d(
                channels,
                channels,
                (kernel_size, kernel_size),
                (1, 1),
                (padding, padding),
                padding_mode='reflect'
            ))

        self.act_fn = create_activation_fn(activation_fn, channels)

    def forward(self, x):
        identity = x
        x = self.conv(x)
        x = self.act_fn(x)

        return x + identity


class HdcDiscriminator(nn.Module):
    def __init__(self, d_depth, image_size, image_channels, score_dim):
        super().__init__()

        activation_fn = 'lrelu'
        padding_mode = 'reflect'

        class FirstBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.fromImage = sn(nn.Conv2d(image_channels, in_channels, (1, 1), (1, 1), (0, 0), bias=False))

                self.compute = ComputeBlock(in_channels, 3, activation_fn)

                self.conv = sn(nn.Conv2d(
                    in_channels,
                    out_channels,
                    (4, 4),
                    (2, 2),
                    (1, 1),
                    padding_mode=padding_mode
                ))

                self.act_fn = create_activation_fn(activation_fn, out_channels)

            def forward(self, x):
                x = self.fromImage(x)

                x = self.compute(x)
                x = self.conv(x)
                x = self.act_fn(x)

                return x

        class IntermediateBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.compute = ComputeBlock(in_channels, 2, activation_fn)

                self.conv = sn(nn.Conv2d(
                    in_channels,
                    out_channels,
                    (4, 4),
                    (2, 2),
                    (1, 1),
                    padding_mode=padding_mode
                ))

                self.act_fn = create_activation_fn(activation_fn, out_channels)

            def forward(self, x):
                x = self.compute(x)
                x = self.conv(x)
                x = self.act_fn(x)

                return x

        class LastBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.compute = ComputeBlock(in_channels, 2, activation_fn)

                self.conv = sn(nn.Conv2d(
                    in_channels,
                    out_channels,
                    (4, 4),
                    (1, 1),
                    (0, 0),
                ))

                self.act_fn = create_activation_fn(activation_fn, out_channels)

                self.reparam = nn.Sequential(
                    Reshape(shape=(-1, score_dim)),
                    sn(nn.Linear(score_dim, score_dim * 2, bias=False)),
                )

            def forward(self, x):
                x = self.compute(x)
                x = self.conv(x)
                x = self.act_fn(x)

                statistics = self.reparam(x)
                mu, log_variance = statistics.chunk(2, dim=1)
                std = log_variance.mul(0.5).exp_()

                epsilon = torch.randn(x.shape[0], score_dim).to(statistics)
                x = epsilon.mul(std).add_(mu)

                return x

        # END block declaration section

        self.channels = [
            *[
                2 ** i * d_depth
                for i in range(1, int(math.log2(image_size)))
            ],
            score_dim
        ]

        self.blocks = nn.ModuleList()

        for i, channel in enumerate(self.channels):
            if i == 0:  # first
                self.blocks.append(
                    FirstBlock(channel, self.channels[i + 1])
                )
            elif 0 < i < len(self.channels) - 2:  # intermediate
                self.blocks.append(
                    IntermediateBlock(channel, self.channels[i + 1])
                )
            elif i < len(self.channels) - 1:  # last
                self.blocks.append(
                    LastBlock(channel, self.channels[i + 1])
                )

    def forward(self, x: Tensor):
        for b in self.blocks:
            x = b(x)

        return x

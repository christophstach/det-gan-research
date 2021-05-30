import math

import torch
from torch import nn, Tensor
from torch.nn.utils import spectral_norm as sn

from layers.reshape import Reshape
from utils import create_activation_fn


class ShuffleDiscriminator(nn.Module):
    def __init__(self, d_depth, image_size, image_channels, score_dim):
        super().__init__()

        bias = True
        activation_fn = 'lrelu'
        padding_mode = 'reflect'

        class FirstBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.conv1 = sn(nn.Conv2d(
                    in_channels,
                    out_channels,
                    (3, 3),
                    (1, 1),
                    (1, 1),
                    padding_mode=padding_mode,
                    bias=bias
                ))

                self.act_fn1 = create_activation_fn(activation_fn, out_channels)

                self.unshuffle = nn.PixelUnshuffle(2)

                self.conv2 = sn(nn.Conv2d(
                    out_channels * 4,
                    out_channels,
                    (3, 3),
                    (1, 1),
                    (1, 1),
                    padding_mode=padding_mode,
                    bias=bias
                ))

                self.act_fn2 = create_activation_fn(activation_fn, out_channels)

            def forward(self, x):
                x = self.conv1(x)
                x = self.act_fn1(x)

                x = self.unshuffle(x)
                x = self.conv2(x)
                x = self.act_fn2(x)

                return x

        class IntermediateBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.conv1 = sn(nn.Conv2d(
                    in_channels,
                    out_channels,
                    (3, 3),
                    (1, 1),
                    (1, 1),
                    padding_mode=padding_mode,
                    bias=bias
                ))

                self.act_fn1 = create_activation_fn(activation_fn, out_channels)

                self.unshuffle = nn.PixelUnshuffle(2)
                self.conv2 = sn(nn.Conv2d(
                    out_channels * 4,
                    out_channels,
                    (3, 3),
                    (1, 1),
                    (1, 1),
                    padding_mode=padding_mode,
                    bias=bias
                ))

                self.act_fn2 = create_activation_fn(activation_fn, out_channels)

            def forward(self, x):
                x = self.conv1(x)
                x = self.act_fn1(x)

                x = self.unshuffle(x)
                x = self.conv2(x)
                x = self.act_fn2(x)

                return x

        class LastBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.conv1 = sn(nn.Conv2d(
                    in_channels,
                    out_channels,
                    (3, 3),
                    (1, 1),
                    (1, 1),
                    padding_mode=padding_mode,
                    bias=bias
                ))

                self.act_fn1 = create_activation_fn(activation_fn, out_channels)

                self.unshuffle = nn.PixelUnshuffle(4)

                self.conv2 = sn(nn.Conv2d(
                    out_channels * 16,
                    out_channels,
                    (1, 1),
                    (1, 1),
                    (0, 0),
                    padding_mode=padding_mode,
                    bias=bias
                ))

                self.act_fn2 = create_activation_fn(activation_fn, out_channels)

                self.reparam = nn.Sequential(
                    Reshape(shape=(-1, out_channels)),
                    sn(nn.Linear(out_channels, out_channels * 2)),
                )

            def forward(self, x):
                x = self.conv1(x)
                x = self.act_fn1(x)

                x = self.unshuffle(x)
                x = self.conv2(x)
                x = self.act_fn2(x)

                statistics = self.reparam(x)
                mu, log_variance = statistics.chunk(2, 1)
                std = log_variance.mul(0.5).exp_()

                epsilon = torch.randn(x.shape[0], score_dim).to(statistics)
                x = epsilon.mul(std).add_(mu)

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

        for i, channel in enumerate(channels):
            if i == 0:  # first
                self.blocks.append(
                    FirstBlock(channel, channels[i + 1])
                )
            elif 0 < i < len(channels) - 2:  # intermediate
                self.blocks.append(
                    IntermediateBlock(channel, channels[i + 1])
                )
            elif i < len(channels) - 1:  # last
                self.blocks.append(
                    LastBlock(channel, channels[i + 1])
                )

        # self.apply(weights_init)

    def forward(self, rgb: Tensor):
        x = rgb

        for b in self.blocks:
            x = b(x)

        return x

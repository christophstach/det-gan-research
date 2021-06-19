import math

import torch
from torch import nn, Tensor
from torch.nn.utils import spectral_norm as sn

from layers.res import DownResBlock
from layers.reshape import Reshape


class ResDiscriminator(nn.Module):
    def __init__(self, d_depth, image_size, image_channels, score_dim, pack=1):

        super().__init__()

        self.pack = pack

        class FirstBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.res = DownResBlock(in_channels, out_channels, first=True)

            def forward(self, x):
                x = self.res(x)

                return x

        class IntermediateBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.res = DownResBlock(in_channels, out_channels)

            def forward(self, x):
                x = self.res(x)

                return x

        class LastBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.res = DownResBlock(in_channels, out_channels, last=True)

                self.reparam = nn.Sequential(
                    Reshape(shape=(-1, score_dim)),
                    sn(nn.Linear(score_dim, score_dim * 2, bias=False)),
                )

            def forward(self, x):
                x = self.res(x)

                statistics = self.reparam(x)
                mu, log_variance = statistics.chunk(2, dim=1)
                std = log_variance.mul(0.5).exp_()

                epsilon = torch.randn(x.shape[0], score_dim).to(statistics)
                x = epsilon.mul(std).add_(mu)

                return x

        # END block declaration section

        self.channels = [
            image_channels,
            *[
                2 ** i * d_depth
                for i in range(2, int(math.log2(image_size)))
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

        def weights_init(m):
            class_name = m.__class__.__name__

            if class_name in ['Conv2d', 'ConvTranspose2d']:
                nn.init.normal_(m.weight.data, 0.0, 0.02)

                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

            elif class_name in ['BatchNorm2d']:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

        self.apply(weights_init)

    def forward(self, x: Tensor):
        if self.pack > 1:
            x = torch.reshape(x, (-1, x.shape[1] * self.pack, x.shape[2], x.shape[3]))

        for b in self.blocks:
            x = b(x)

        return x

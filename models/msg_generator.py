import math

import torch
from torch import nn, Tensor

from utils import create_activation_fn, create_norm, create_upscale


class MsgGenerator(nn.Module):
    def __init__(self, g_depth, image_size, image_channels, latent_dim):
        super().__init__()

        norm = 'pixel'
        activation_fn = 'lrelu'
        upscale = 'interpolate'

        class FirstBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.compute1 = nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, (4, 4), (1, 1), (0, 0)),
                    create_activation_fn(activation_fn, out_channels)
                )

                self.compute2 = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), (1, 1)),
                    create_activation_fn(activation_fn, out_channels)
                )

                self.norm1 = create_norm(norm, out_channels)
                self.norm2 = create_norm(norm, out_channels)

                self.noise1 = nn.Parameter(Tensor(out_channels, 1, 1).fill_(1.0))
                self.noise2 = nn.Parameter(Tensor(out_channels, 1, 1).fill_(1.0))

                self.toRGB = nn.Sequential(
                    nn.Conv2d(out_channels, image_channels, (1, 1), (1, 1), (0, 0)),
                    nn.Tanh()
                )

            def forward(self, x):
                x = self.compute1(x)
                x += torch.randn_like(x) * self.noise1
                x = self.norm1(x)

                x = self.compute2(x)
                x += torch.randn_like(x) * self.noise2
                x = self.norm2(x)

                return x, self.toRGB(x)

        class IntermediateBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.compute1 = nn.Sequential(
                    create_upscale(upscale, in_channels),
                    nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1)),
                    create_activation_fn(activation_fn, out_channels)
                )

                self.compute2 = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), (1, 1)),
                    create_activation_fn(activation_fn, out_channels)
                )

                self.norm1 = create_norm(norm, out_channels)
                self.norm2 = create_norm(norm, out_channels)

                self.noise1 = nn.Parameter(Tensor(out_channels, 1, 1).fill_(1.0))
                self.noise2 = nn.Parameter(Tensor(out_channels, 1, 1).fill_(1.0))

                self.toRGB = nn.Sequential(
                    nn.Conv2d(out_channels, image_channels, (1, 1), (1, 1), (0, 0)),
                    nn.Tanh()
                )

            def forward(self, x):
                x = self.compute1(x)
                x += torch.randn_like(x) * self.noise1
                x = self.norm1(x)

                x = self.compute2(x)
                x += torch.randn_like(x) * self.noise2
                x = self.norm2(x)

                return x, self.toRGB(x)

        class LastBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.compute1 = nn.Sequential(
                    create_upscale(upscale, in_channels),
                    nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1)),
                    create_activation_fn(activation_fn, out_channels)
                )

                self.compute2 = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), (1, 1)),
                    create_activation_fn(activation_fn, out_channels)
                )

                self.norm1 = create_norm(norm, out_channels)
                self.norm2 = create_norm(norm, out_channels)

                self.noise1 = nn.Parameter(Tensor(out_channels, 1, 1).fill_(1.0))
                self.noise2 = nn.Parameter(Tensor(out_channels, 1, 1).fill_(1.0))

                self.toRGB = nn.Sequential(
                    nn.Conv2d(out_channels, image_channels, (1, 1), (1, 1), (0, 0)),
                    nn.Tanh()
                )

            def forward(self, x):
                x = self.compute1(x)
                x += torch.randn_like(x) * self.noise1
                x = self.norm1(x)

                x = self.compute2(x)
                x += torch.randn_like(x) * self.noise2
                x = self.norm2(x)

                return x, self.toRGB(x)

        # END block declaration section

        self.channels = [
            latent_dim,
            *list(reversed([
                2 ** i * g_depth
                for i in range(2, int(math.log2(image_size)))
            ])),
            image_channels
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

    def forward(self, x):
        rgbs = []
        for b in self.blocks:
            x, rgb = b(x)
            rgbs.insert(0, rgb)

        return rgbs

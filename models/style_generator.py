import math

import torch
from torch import nn, Tensor

from layers.adain import AdaptiveInstanceNormalization2d
from utils import create_activation_fn, create_upscale


class StyleGenerator(nn.Module):
    def __init__(self, g_depth, image_size, image_channels, latent_dim):
        super().__init__()

        activation_fn = 'lrelu'
        disentangler_activation_fn = 'mish'
        upscale = 'nearest'

        class Conv(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.conv = nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1), padding_mode='replicate')

            def forward(self, x):
                return self.conv(x)

        class FirstBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.compute1 = nn.Sequential(
                    Conv(in_channels, out_channels),
                    create_activation_fn(activation_fn, out_channels)
                )

                self.compute2 = nn.Sequential(
                    Conv(out_channels, out_channels),
                    create_activation_fn(activation_fn, out_channels)
                )

                self.norm1 = AdaptiveInstanceNormalization2d(latent_dim, out_channels)
                self.norm2 = AdaptiveInstanceNormalization2d(latent_dim, out_channels)

                self.noise1 = nn.Parameter(Tensor(out_channels, 1, 1).fill_(1.0))
                self.noise2 = nn.Parameter(Tensor(out_channels, 1, 1).fill_(1.0))

                self.toRGB = nn.Sequential(
                    nn.Conv2d(out_channels, image_channels, (1, 1), (1, 1), (0, 0)),
                    nn.Tanh()
                )

            def forward(self, x, w):
                x = self.compute1(x)
                # x += torch.randn_like(x) * self.noise1
                x = self.norm1(x, w)

                x = self.compute2(x)
                # x += torch.randn_like(x) * self.noise2
                x = self.norm2(x, w)

                return x, self.toRGB(x)

        class IntermediateBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.compute1 = nn.Sequential(
                    create_upscale(upscale, in_channels),
                    Conv(in_channels, out_channels),
                    create_activation_fn(activation_fn, out_channels),
                )

                self.compute2 = nn.Sequential(
                    Conv(out_channels, out_channels),
                    create_activation_fn(activation_fn, out_channels),
                )

                self.norm1 = AdaptiveInstanceNormalization2d(latent_dim, out_channels)
                self.norm2 = AdaptiveInstanceNormalization2d(latent_dim, out_channels)

                self.noise1 = nn.Parameter(Tensor(out_channels, 1, 1).fill_(1.0))
                self.noise2 = nn.Parameter(Tensor(out_channels, 1, 1).fill_(1.0))

                self.toRGB = nn.Sequential(
                    nn.Conv2d(out_channels, image_channels, (1, 1), (1, 1), (0, 0)),
                    nn.Tanh()
                )

            def forward(self, x, w):
                x = self.compute1(x)
                # x += torch.randn_like(x) * self.noise1
                x = self.norm1(x, w)

                x = self.compute2(x)
                # x += torch.randn_like(x) * self.noise2
                x = self.norm2(x, w)

                return x, self.toRGB(x)

        class LastBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.compute1 = nn.Sequential(
                    create_upscale(upscale, in_channels),
                    Conv(in_channels, out_channels),
                    create_activation_fn(activation_fn, out_channels)
                )

                self.compute2 = nn.Sequential(
                    Conv(out_channels, out_channels),
                    create_activation_fn(activation_fn, out_channels)
                )

                self.norm1 = AdaptiveInstanceNormalization2d(latent_dim, out_channels)
                self.norm2 = AdaptiveInstanceNormalization2d(latent_dim, out_channels)

                self.noise1 = nn.Parameter(Tensor(out_channels, 1, 1).fill_(1.0))
                self.noise2 = nn.Parameter(Tensor(out_channels, 1, 1).fill_(1.0))

                self.toRGB = nn.Sequential(
                    nn.Conv2d(out_channels, image_channels, (1, 1), (1, 1), (0, 0)),
                    nn.Tanh()
                )

            def forward(self, x, w):
                x = self.compute1(x)
                # x += torch.randn_like(x) * self.noise1
                x = self.norm1(x, w)

                x = self.compute2(x)
                # x += torch.randn_like(x) * self.noise2
                x = self.norm2(x, w)

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

        self.disentangler = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, (1, 1), (1, 1), (0, 0)),
            create_activation_fn(disentangler_activation_fn, latent_dim),
            nn.Conv2d(latent_dim, latent_dim, (1, 1), (1, 1), (0, 0)),
            create_activation_fn(disentangler_activation_fn, latent_dim),
            nn.Conv2d(latent_dim, latent_dim, (1, 1), (1, 1), (0, 0)),
            create_activation_fn(disentangler_activation_fn, latent_dim)
        )

        self.const = nn.Parameter(Tensor(latent_dim, 4, 4))
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

    def forward(self, z):
        rgbs = []

        w = self.disentangler(z)
        x = torch.unsqueeze(self.const, dim=0).repeat(z.shape[0], 1, 1, 1)

        for b in self.blocks:
            x, rgb = b(x, w)
            rgbs.insert(0, rgb)

        return rgbs

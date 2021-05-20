import math
import torch
from torch import nn

from layers.adain import AdaptiveInstanceNormalization2d
from layers.eql import EqlConv2d
from layers.mod import ModConv
from utils import create_activation_fn, create_upscale
from torch.nn.utils import spectral_norm as sn

class StyleMsgGenerator(nn.Module):
    def __init__(self, g_depth, image_size, image_channels, latent_dim):
        super().__init__()

        activation_fn = 'lrelu'
        upscale = 'bilinear'
        disentangler_activation_fn = 'lrelu'
        eql = False

        class Conv(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                if eql:
                    self.conv = EqlConv2d(
                        in_channels,
                        out_channels,
                        (3, 3),
                        (1, 1),
                        (1, 1),
                        padding_mode='reflect'
                    )

                else:
                    self.conv = nn.Conv2d(
                        in_channels,
                        out_channels,
                        (3, 3),
                        (1, 1),
                        (1, 1),
                        padding_mode='reflect'
                    )

            def forward(self, x):
                return self.conv(x)

        class FirstBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.compute1 = sn(ModConv(in_channels, out_channels, latent_dim))
                self.compute2 = sn(ModConv(out_channels, out_channels, latent_dim))

                # self.norm1 = AdaptiveInstanceNormalization2d(latent_dim, out_channels)
                # self.norm2 = AdaptiveInstanceNormalization2d(latent_dim, out_channels)

                self.act_fn1 = create_activation_fn(activation_fn, out_channels)
                self.act_fn2 = create_activation_fn(activation_fn, out_channels)

                if eql:
                    self.toRGB = EqlConv2d(out_channels, image_channels, (1, 1), (1, 1), (0, 0))
                else:
                    self.toRGB = nn.Conv2d(out_channels, image_channels, (1, 1), (1, 1), (0, 0))

            def forward(self, x, w):
                x = self.compute1(x, w)
                # x = self.compute1(x)
                # x = self.norm1(x, w)
                x = self.act_fn1(x)

                x = self.compute2(x, w)
                # x = self.compute2(x)
                # x = self.norm2(x, w)
                x = self.act_fn2(x)

                rgb = self.toRGB(x)
                return x, rgb

        class IntermediateBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.up = create_upscale(upscale)

                self.compute1 = sn(ModConv(in_channels, out_channels, latent_dim))
                self.compute2 = sn(ModConv(out_channels, out_channels, latent_dim))

                # self.norm1 = AdaptiveInstanceNormalization2d(latent_dim, out_channels)
                # self.norm2 = AdaptiveInstanceNormalization2d(latent_dim, out_channels)

                self.act_fn1 = create_activation_fn(activation_fn, out_channels)
                self.act_fn2 = create_activation_fn(activation_fn, out_channels)

                if eql:
                    self.toRGB = EqlConv2d(out_channels, image_channels, (1, 1), (1, 1), (0, 0))
                else:
                    self.toRGB = nn.Conv2d(out_channels, image_channels, (1, 1), (1, 1), (0, 0))

            def forward(self, x, w):
                x = self.up(x)

                x = self.compute1(x, w)
                # x = self.compute1(x)
                # x = self.norm1(x, w)
                x = self.act_fn1(x)

                x = self.compute2(x, w)
                # x = self.compute2(x)
                # x = self.norm2(x, w)
                x = self.act_fn2(x)

                rgb = self.toRGB(x)
                return x, rgb

        class LastBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.up = create_upscale(upscale)

                self.compute1 = sn(ModConv(in_channels, out_channels, latent_dim))
                self.compute2 = sn(ModConv(out_channels, out_channels, latent_dim))

                # self.norm1 = AdaptiveInstanceNormalization2d(latent_dim, out_channels)
                # self.norm2 = AdaptiveInstanceNormalization2d(latent_dim, out_channels)

                self.act_fn1 = create_activation_fn(activation_fn, out_channels)
                self.act_fn2 = create_activation_fn(activation_fn, out_channels)

                if eql:
                    self.toRGB = EqlConv2d(out_channels, image_channels, (1, 1), (1, 1), (0, 0))
                else:
                    self.toRGB = nn.Conv2d(out_channels, image_channels, (1, 1), (1, 1), (0, 0))

            def forward(self, x, w):
                x = self.up(x)

                x = self.compute1(x, w)
                # x = self.compute1(x)
                # x = self.norm1(x, w)
                x = self.act_fn1(x)

                x = self.compute2(x, w)
                # x = self.compute2(x)
                # x = self.norm2(x, w)
                x = self.act_fn2(x)

                rgb = self.toRGB(x)
                return x, rgb

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
            nn.Linear(latent_dim, latent_dim),
            create_activation_fn(disentangler_activation_fn, latent_dim),
            nn.Linear(latent_dim, latent_dim),
            create_activation_fn(disentangler_activation_fn, latent_dim),
            nn.Linear(latent_dim, latent_dim),
            create_activation_fn(disentangler_activation_fn, latent_dim)
        )

        self.const = nn.Parameter(
            nn.init.normal_(
                torch.empty(latent_dim, 4, 4)
            )
        )

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
        # w = self.disentangler(z).view(*z.shape, 1, 1)
        w = self.disentangler(z)
        x = torch.unsqueeze(self.const, dim=0).repeat(z.shape[0], 1, 1, 1)

        rgbs = []
        for b in self.blocks:
            x, rgb = b(x, w)
            rgbs.insert(0, rgb)

        return rgbs

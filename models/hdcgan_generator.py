import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from layers.adain import AdaptiveInstanceNormalization2d
from layers.reshape import Reshape


class HdcGenerator(nn.Module):
    def __init__(self, g_depth, image_size, image_channels, latent_dim):
        super().__init__()

        class FirstBlock(nn.Module):
            def __init__(self, in_channels, out_channels, style_dim):
                super().__init__()

                self.reshape = Reshape(shape=(-1, in_channels, 1, 1))

                self.up = nn.ConvTranspose2d(
                    in_channels,
                    in_channels,
                    kernel_size=(4, 4)
                )

                self.conv = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(3, 3),
                    padding=(1, 1),
                    padding_mode='reflect'
                )

                self.adain = AdaptiveInstanceNormalization2d(style_dim, in_channels)
                self.bn = nn.BatchNorm2d(out_channels)

            def forward(self, x, w):
                x = self.reshape(x)

                x = self.up(x)
                x = self.adain(x, w)
                x = F.leaky_relu(x, 0.2, inplace=True)

                x = self.conv(x)
                x = self.bn(x)
                x = F.leaky_relu(x, 0.2, inplace=True)

                return x

        class IntermediateBlock(nn.Module):
            def __init__(self, in_channels, out_channels, style_dim):
                super().__init__()

                self.up = nn.ConvTranspose2d(
                    in_channels,
                    in_channels,
                    kernel_size=(4, 4),
                    stride=(2, 2),
                    padding=(1, 1)
                )

                self.conv = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(3, 3),
                    padding=(1, 1),
                    padding_mode='reflect'
                )

                self.adain = AdaptiveInstanceNormalization2d(style_dim, in_channels)
                self.bn = nn.BatchNorm2d(out_channels)

            def forward(self, x, w):
                x = self.up(x)
                x = self.adain(x, w)
                x = F.leaky_relu(x, 0.2, inplace=True)

                x = self.conv(x)
                x = self.bn(x)
                x = F.leaky_relu(x, 0.2, inplace=True)

                return x

        class LastBlock(nn.Module):
            def __init__(self, in_channels, out_channels, style_dim):
                super().__init__()

                self.up = nn.ConvTranspose2d(
                    in_channels,
                    in_channels,
                    kernel_size=(4, 4),
                    stride=(2, 2),
                    padding=(1, 1)
                )

                self.conv = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(3, 3),
                    padding=(1, 1),
                    padding_mode='reflect'
                )

                self.adain = AdaptiveInstanceNormalization2d(style_dim, in_channels)

            def forward(self, x, w):
                x = self.up(x)
                x = self.adain(x, w)
                x = F.leaky_relu(x, 0.2, inplace=True)

                x = self.conv(x)
                x = torch.tanh(x)

                return x

        # END block declaration section

        self.split_latent_space = False

        if self.split_latent_space:
            self.latent_splits = self.calculate_latent_splits(latent_dim, int(math.log2(image_size)) - 1)

        self.channels = [
            self.latent_splits[0] if self.split_latent_space else latent_dim,
            *list(reversed([
                2 ** i * g_depth
                for i in range(2, int(math.log2(image_size)))
            ])),
            image_channels
        ]

        if not self.split_latent_space:
            self.const = nn.Parameter(
                nn.init.normal_(
                    torch.empty(self.channels[0])
                )
            )

        self.blocks = nn.ModuleList()

        for i, channel in enumerate(self.channels):
            if i == 0:  # first
                self.blocks.append(
                    FirstBlock(
                        channel,
                        self.channels[i + 1],
                        self.latent_splits[i + 1] if self.split_latent_space else latent_dim
                    )
                )
            elif 0 < i < len(self.channels) - 2:  # intermediate
                self.blocks.append(
                    IntermediateBlock(
                        channel,
                        self.channels[i + 1],
                        self.latent_splits[i + 1] if self.split_latent_space else latent_dim
                    )
                )
            elif i < len(self.channels) - 1:  # last
                self.blocks.append(
                    LastBlock(
                        channel,
                        self.channels[i + 1],
                        self.latent_splits[i + 1] if self.split_latent_space else latent_dim
                    )
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

    @staticmethod
    def calculate_latent_splits(latent_dim, n_blocks):
        closest_power_of_two = 1
        max_divisible = math.ceil(latent_dim / (n_blocks + 1))

        while True:
            if closest_power_of_two > max_divisible:
                break
            else:
                closest_power_of_two *= 2

        closest_power_of_two //= 2

        splits = [closest_power_of_two for _ in range(n_blocks)]
        splits.insert(0, latent_dim - (n_blocks * closest_power_of_two))

        return splits

    def forward(self, z: Tensor):
        w = z.view(z.shape[0], -1, 1, 1)

        if self.split_latent_space:
            w = torch.split(w, self.latent_splits, dim=1)
            x = w[0]

            for i, b in enumerate(self.blocks):
                x = b(x, w[i + 1])
        else:
            x = torch.unsqueeze(self.const, dim=0).repeat(z.shape[0], 1)

            for b in self.blocks:
                x = b(x, w)

        return x, w

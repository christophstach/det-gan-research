import math

import torch
from torch import nn, Tensor

from layers.adain import AdaptiveInstanceNormalization2d
from layers.pad import EvenPad2d
from utils import create_activation_fn, create_norm


class ComputeBlock(nn.Module):
    def __init__(self, channels, kernel_size: int, norm: str, activation_fn: str):
        super().__init__()

        if kernel_size % 2 == 0:
            self.conv = nn.Sequential(
                EvenPad2d(kernel_size, 'reflect'),
                nn.Conv2d(channels, channels, (kernel_size, kernel_size), (1, 1), (0, 0))
            )
        else:
            padding = kernel_size // 2
            self.conv = nn.Conv2d(
                channels,
                channels,
                (kernel_size, kernel_size),
                (1, 1),
                (padding, padding),
                padding_mode='reflect'
            )

        self.norm = create_norm(norm, channels)

        self.act_fn = create_activation_fn(activation_fn, channels)

    def forward(self, x):
        identity = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.act_fn(x)

        return x + identity


class HdcGenerator(nn.Module):
    def __init__(self, g_depth, image_size, image_channels, latent_dim):
        super().__init__()

        norm = 'batch'
        activation_fn = 'lrelu'
        disentangler_activation_fn = 'mish'

        class FirstBlock(nn.Module):
            def __init__(self, in_channels, out_channels, style_dim):
                super().__init__()

                self.conv = nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    (4, 4),
                    (1, 1),
                    (0, 0)
                )

                self.norm = AdaptiveInstanceNormalization2d(style_dim, out_channels)
                self.act_fn = create_activation_fn(activation_fn, out_channels)
                self.compute = ComputeBlock(out_channels, 2, norm, activation_fn)

            def forward(self, x, w):
                x = self.conv(x)
                x = self.norm(x, w)
                x = self.act_fn(x)
                x = self.compute(x)

                return x

        class IntermediateBlock(nn.Module):
            def __init__(self, in_channels, out_channels, style_dim):
                super().__init__()

                self.conv = nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    (4, 4),
                    (2, 2),
                    (1, 1)
                )

                self.norm = AdaptiveInstanceNormalization2d(style_dim, out_channels)
                self.act_fn = create_activation_fn(activation_fn, out_channels)
                self.compute = ComputeBlock(out_channels, 2, norm, activation_fn)

            def forward(self, x, w):
                x = self.conv(x)
                x = self.norm(x, w)
                x = self.act_fn(x)
                x = self.compute(x)

                return x

        class LastBlock(nn.Module):
            def __init__(self, in_channels, out_channels, style_dim):
                super().__init__()

                self.conv = nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    (4, 4),
                    (2, 2),
                    (1, 1)
                )

                self.norm = AdaptiveInstanceNormalization2d(style_dim, out_channels)
                self.act_fn = create_activation_fn(activation_fn, out_channels)
                self.compute = ComputeBlock(out_channels, 3, norm, activation_fn)

                self.toImage = nn.Sequential(
                    nn.Conv2d(out_channels, image_channels, (1, 1), (1, 1), (0, 0), bias=False),
                    nn.Tanh()
                )

            def forward(self, x, w):
                x = self.conv(x)
                x = self.norm(x, w)
                x = self.act_fn(x)
                x = self.compute(x)

                x = self.toImage(x)

                return x

        # END block declaration section

        self.split_latent_space = True

        if self.split_latent_space:
            self.latent_splits = self.calculate_latent_splits(latent_dim, int(math.log2(image_size)) - 1)

        self.channels = [
            self.latent_splits[0] if self.split_latent_space else latent_dim,
            *list(reversed([
                2 ** i * g_depth
                for i in range(1, int(math.log2(image_size)))
            ]))
        ]

        self.disentangler = nn.Sequential(
            # nn.Conv2d(latent_dim, latent_dim, (1, 1), (1, 1), (0, 0), bias=bias),
            # create_activation_fn(disentangler_activation_fn, latent_dim)
        )

        if not self.split_latent_space:
            self.const = nn.Parameter(
                nn.init.normal_(
                    torch.empty(self.channels[0], 1, 1)
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

    def forward(self, z: Tensor):
        w = self.disentangler(z.view(z.shape[0], -1, 1, 1))

        if self.split_latent_space:
            w = torch.split(w, self.latent_splits, dim=1)
            x = w[0]

            for i, b in enumerate(self.blocks):
                x = b(x, w[i + 1])
        else:
            x = torch.unsqueeze(self.const, dim=0).repeat(z.shape[0], 1, 1, 1)

            for b in self.blocks:
                x = b(x, w)

        return x, w

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

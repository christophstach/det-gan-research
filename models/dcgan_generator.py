import math
import torch
from torch import nn, Tensor

from utils import create_activation_fn, create_norm


class DcGenerator(nn.Module):
    def __init__(self, g_depth, image_size, image_channels, latent_dim):
        super().__init__()

        bias = False
        norm = 'batch'
        activation_fn = 'relu'
        from_w_activation_fn = 'mish'
        disentangler_activation_fn = 'mish'
        padding_mode = 'zeros'

        def weights_init(m):
            class_name = m.__class__.__name__
            if class_name.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif class_name.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

        class FirstBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.fromW = nn.Sequential(
                    nn.Conv2d(latent_dim, in_channels, (1, 1), (1, 1), (0, 0), bias=bias),
                    create_activation_fn(from_w_activation_fn, latent_dim)
                )

                self.compute = nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    (4, 4),
                    (1, 1),
                    (0, 0),
                    padding_mode=padding_mode,
                    bias=bias
                )

                self.norm = create_norm(norm, out_channels)
                self.act_fn = create_activation_fn(activation_fn, out_channels)

            def forward(self, x, w):
                x = x + self.fromW(w)

                x = self.compute(x)
                x = self.norm(x)
                x = self.act_fn(x)

                return x

        class IntermediateBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.fromW = nn.Sequential(
                    nn.Conv2d(latent_dim, in_channels, (1, 1), (1, 1), (0, 0), bias=bias),
                    create_activation_fn(from_w_activation_fn, latent_dim)
                )

                self.compute = nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    (4, 4),
                    (2, 2),
                    (1, 1),
                    padding_mode=padding_mode,
                    bias=bias
                )

                self.norm = create_norm(norm, out_channels)
                self.act_fn = create_activation_fn(activation_fn, out_channels)

            def forward(self, x, w):
                x = x + self.fromW(w)

                x = self.compute(x)
                x = self.norm(x)
                x = self.act_fn(x)

                return x

        class LastBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.fromW = nn.Sequential(
                    nn.Conv2d(latent_dim, in_channels, (1, 1), (1, 1), (0, 0), bias=bias),
                    create_activation_fn(from_w_activation_fn, latent_dim)
                )

                self.compute = nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    (4, 4),
                    (2, 2),
                    (1, 1),
                    padding_mode=padding_mode,
                    bias=bias
                )

            def forward(self, x, w):
                x = x + self.fromW(w)

                x = self.compute(x)
                x = torch.tanh(x)

                return x

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
            nn.Conv2d(latent_dim, latent_dim, (1, 1), (1, 1), (0, 0), bias=bias),
            create_activation_fn(disentangler_activation_fn, latent_dim)
        )

        self.const = nn.Parameter(
            nn.init.normal_(
                torch.empty(latent_dim),
                0.0,
                0.02
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

        self.apply(weights_init)

    def forward(self, z: Tensor):
        w = self.disentangler(z.view(z.shape[0], -1, 1, 1))

        x = torch.unsqueeze(self.const, dim=0).repeat(z.shape[0], 1)
        x = x.view(z.shape[0], -1, 1, 1)

        for b in self.blocks:
            x = b(x, w)

        return x

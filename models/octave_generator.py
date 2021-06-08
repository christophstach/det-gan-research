import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from layers.octave import OctaveConv


class OctaveGenerator(nn.Module):
    def __init__(self, g_depth, image_size, image_channels, latent_dim):
        super().__init__()

        class FirstBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.conv = nn.ConvTranspose2d(
                    in_channels,
                    out_channels // 2,
                    (4, 4),
                    (1, 1),
                    (0, 0),
                    bias=False
                )

                self.norm = nn.BatchNorm2d(out_channels // 2)

            def forward(self, x):
                x = self.conv(x)
                x = self.norm(x)
                x = F.relu(x, inplace=True)

                return x

        class IntermediateBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.conv = OctaveConv(in_channels, out_channels, (3, 3))

                self.norm_h = nn.BatchNorm2d(out_channels)
                self.norm_l = nn.BatchNorm2d(out_channels)

            def forward(self, x):
                print(x.shape)
                x_h, x_l = self.conv(x)
                print(x_h.shape)
                print(x_l.shape)

                x_h = self.norm_h(x_h)
                x_l = self.norm_h(x_l)

                x_h = F.relu(x_h, inplace=True)
                x_l = F.relu(x_l, inplace=True)

                return x_h, x_l

        class LastBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.conv = nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    (4, 4),
                    (2, 2),
                    (1, 1),
                    bias=False
                )

            def forward(self, x):
                x = self.conv(x)
                x = torch.tanh(x)

                return x

        # END block declaration section

        self.channels = [
            latent_dim,
            *list(reversed([
                2 ** i * g_depth
                for i in range(1, int(math.log2(image_size)))
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

    def forward(self, z: Tensor):
        x = z.view(z.shape[0], -1, 1, 1)

        for b in self.blocks:
            x = b(x)

        return x

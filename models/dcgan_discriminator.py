import math
import torch
from torch import nn, Tensor
from torch.nn.utils import spectral_norm as sn

from layers.reshape import Reshape
from utils import create_activation_fn, create_norm


class DcDiscriminator(nn.Module):
    def __init__(self, d_depth, image_size, image_channels, score_dim):
        super().__init__()

        bias = False
        norm = 'passthrough'
        activation_fn = 'lrelu'
        padding_mode = 'reflect'

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

                self.compute = sn(nn.Conv2d(
                    in_channels,
                    out_channels,
                    (4, 4),
                    (2, 2),
                    (1, 1),
                    padding_mode=padding_mode,
                    bias=bias
                ))

                self.norm = create_norm(norm, out_channels)
                self.act_fn = create_activation_fn(activation_fn, out_channels)

            def forward(self, x):
                x = self.compute(x)
                x = self.norm(x)
                x = self.act_fn(x)

                return x

        class IntermediateBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.compute = sn(nn.Conv2d(
                    in_channels,
                    out_channels,
                    (4, 4),
                    (2, 2),
                    (1, 1),
                    padding_mode=padding_mode,
                    bias=bias
                ))

                self.norm = create_norm(norm, out_channels)
                self.act_fn = create_activation_fn(activation_fn, out_channels)

            def forward(self, x):
                x = self.compute(x)
                x = self.norm(x)
                x = self.act_fn(x)

                return x

        class LastBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.compute = sn(nn.Conv2d(
                    in_channels,
                    out_channels,
                    (4, 4),
                    (1, 1),
                    (0, 0),
                    padding_mode=padding_mode,
                    bias=bias
                ))

                self.reparam = nn.Sequential(
                    Reshape(shape=(-1, out_channels)),
                    sn(nn.Linear(out_channels, out_channels * 2)),
                )

            def forward(self, x):
                x = self.compute(x)

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

        self.apply(weights_init)

    def forward(self, rgb: Tensor):
        x = rgb

        for b in self.blocks:
            x = b(x)

        return x

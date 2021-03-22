import torch.nn as nn
import torch.nn.functional as F

import utils


class GeneratorFirstBlock(nn.Module):
    def __init__(self, in_channels, norm, activation_fn, latent_dimension, z_skip=True, bias=True):
        super().__init__()

        self.z_skip = z_skip

        if z_skip:
            self.skipper = nn.Conv2d(
                in_channels=latent_dimension,
                out_channels=in_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0)
            )

        self.conv1 = nn.ConvTranspose2d(
            in_channels,
            in_channels,
            kernel_size=(4, 4),
            stride=(1, 1),
            padding=(0, 0),
            bias=bias
        )

        self.conv2 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=bias
        )

        self.act_fn1 = utils.create_activation_fn(activation_fn, in_channels)
        self.act_fn2 = utils.create_activation_fn(activation_fn, in_channels)

        self.norm1 = utils.create_norm(norm, in_channels)
        self.norm2 = utils.create_norm(norm, in_channels)

    def forward(self, x, skip=None):
        x = self.norm1(self.act_fn1(self.conv1(x)))
        x = x + self.skipper(skip) if self.z_skip else x
        x = self.norm2(self.act_fn2(self.conv2(x)))

        return x


class GeneratorIntermediateBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm, activation_fn, latent_dimension, z_skip=True, bias=True):
        super().__init__()

        self.z_skip = z_skip

        if z_skip:
            self.skipper = nn.Conv2d(
                in_channels=latent_dimension,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0)
            )

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=bias
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=bias
        )

        self.act_fn1 = utils.create_activation_fn(activation_fn, out_channels)
        self.act_fn2 = utils.create_activation_fn(activation_fn, out_channels)

        self.norm1 = utils.create_norm(norm, out_channels)
        self.norm2 = utils.create_norm(norm, out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(
            x,
            size=(
                x.size(2) * 2,
                x.size(3) * 2
            ),
            mode="bilinear",
            align_corners=False
        )

        x = self.norm1(self.act_fn1(self.conv1(x)))
        x = x + self.skipper(skip) if self.z_skip else x
        x = self.norm2(self.act_fn2(self.conv2(x)))

        return x


class GeneratorLastBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm, activation_fn, latent_dimension, z_skip=True, bias=False):
        super().__init__()

        self.z_skip = z_skip

        if z_skip:
            self.skipper = nn.Conv2d(
                in_channels=latent_dimension,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0)
            )

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=bias
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=bias
        )

        self.act_fn1 = utils.create_activation_fn(activation_fn, out_channels)
        self.act_fn2 = utils.create_activation_fn(activation_fn, out_channels)

        self.norm1 = utils.create_norm(norm, out_channels)
        self.norm2 = utils.create_norm(norm, out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(
            x,
            size=(
                x.size(2) * 2,
                x.size(3) * 2
            ),
            mode="bilinear",
            align_corners=False
        )

        x = self.norm1(self.act_fn1(self.conv1(x)))
        x = x + self.skipper(skip) if self.z_skip else x
        x = self.norm2(self.act_fn2(self.conv2(x)))

        return x

import torch.nn as nn
import torch.nn.functional as F

import utils


class GeneratorFirstBlock(nn.Module):
    def __init__(self, in_channels, norm, activation_fn, bias=False):
        super().__init__()

        self.conv1 = nn.ConvTranspose2d(
            in_channels,
            in_channels,
            kernel_size=4,
            stride=1,
            padding=0,
            bias=bias
        )

        self.conv2 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias
        )

        self.act_fn1 = utils.create_activation_fn(activation_fn, in_channels)
        self.act_fn2 = utils.create_activation_fn(activation_fn, in_channels)

        self.norm1 = utils.create_norm(norm, in_channels)
        self.norm2 = utils.create_norm(norm, in_channels)

    def forward(self, x):
        x = self.norm1(self.act_fn1(self.conv1(x)))
        x = self.norm2(self.act_fn2(self.conv2(x)))

        return x


class GeneratorIntermediateBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm, activation_fn, bias=False):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias
        )

        self.act_fn1 = utils.create_activation_fn(activation_fn, out_channels)
        self.act_fn2 = utils.create_activation_fn(activation_fn, out_channels)

        self.norm1 = utils.create_norm(norm, out_channels)
        self.norm2 = utils.create_norm(norm, out_channels)

    def forward(self, x):
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
        x = self.norm2(self.act_fn2(self.conv2(x)))

        return x


class GeneratorLastBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm, activation_fn, bias=False):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias
        )

        self.act_fn1 = utils.create_activation_fn(activation_fn, out_channels)
        self.act_fn2 = utils.create_activation_fn(activation_fn, out_channels)

        self.norm1 = utils.create_norm(norm, out_channels)
        self.norm2 = utils.create_norm(norm, out_channels)

    def forward(self, x):
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
        x = self.norm2(self.act_fn2(self.conv2(x)))

        return x

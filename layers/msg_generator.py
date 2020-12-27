import torch.nn as nn
import torch.nn.functional as F

import utils


class MsgGeneratorFirstBlock(nn.Module):
    def __init__(self, in_channels, norm, bias=False):
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

        self.leakyRelu = nn.LeakyReLU(0.2, inplace=True)
        self.norm = utils.create_norm(norm, in_channels)

    def forward(self, x):
        x = self.leakyRelu(self.conv1(x))
        x = self.leakyRelu(self.conv2(x))

        x = self.norm(x)

        return x


class MsgGeneratorIntermediateBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm, bias=False):
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

        self.leakyRelu = nn.LeakyReLU(0.2, inplace=True)

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

        x = self.norm1(self.leakyRelu(self.conv1(x)))
        x = self.norm2(self.leakyRelu(self.conv2(x)))

        return x


class MsgGeneratorLastBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm, bias=False):
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

        self.leakyRelu = nn.LeakyReLU(0.2, inplace=True)

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

        x = self.norm1(self.leakyRelu(self.conv1(x)))
        x = self.norm2(self.leakyRelu(self.conv2(x)))

        return x

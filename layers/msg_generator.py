import torch.nn as nn
import torch.nn.functional as F

import layers as l


class MsgGeneratorFirstBlock(nn.Module):
    def __init__(self, in_channels, bias=False):
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
        self.pixelNorm = l.PixelNorm()

    def forward(self, x):
        x = self.leakyRelu(self.conv1(x))
        x = self.leakyRelu(self.conv2(x))

        x = self.pixelNorm(x)

        return x


class MsgGeneratorIntermediateBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
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
        self.pixelNorm = l.PixelNorm()

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

        x = self.pixelNorm(self.leakyRelu(self.conv1(x)))
        x = self.pixelNorm(self.leakyRelu(self.conv2(x)))

        return x


class MsgGeneratorLastBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
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
        self.pixelNorm = l.PixelNorm()

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

        x = self.pixelNorm(self.leakyRelu(self.conv1(x)))
        x = self.pixelNorm(self.leakyRelu(self.conv2(x)))

        return x

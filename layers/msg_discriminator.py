import torch
import torch.nn as nn
import torch.nn.functional as F

import layers as l


class MsgDiscriminatorFirstBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias
        )

        self.conv2 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias
        )

        self.avgPool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        x = self.conv1(x)
        F.leaky_relu(x, 0.2, inplace=True)

        x = self.conv2(x)
        F.leaky_relu(x, 0.2, inplace=True)

        x = self.avgPool(x)

        return x


class MsgDiscriminatorIntermediateBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias
        )

        self.conv2 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias
        )

        self.avgPool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        x = self.conv1(x)
        F.leaky_relu(x, 0.2, inplace=True)

        x = self.conv2(x)
        F.leaky_relu(x, 0.2, inplace=True)

        x = self.avgPool(x)

        return x


class MsgDiscriminatorLastBlock(nn.Module):
    def __init__(self, in_channels, bias=False):
        super().__init__()

        self.miniBatchStdDev = l.MinibatchStdDev()

        self.conv1 = nn.Conv2d(
            in_channels + 1,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias
        )

        self.conv2 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=4,
            stride=1,
            padding=0,
            bias=bias
        )

        self.validator = nn.Conv2d(
            in_channels,
            1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )

    def forward(self, x):
        x = self.miniBatchStdDev(x)

        x = self.conv1(x)
        F.leaky_relu(x, 0.2, inplace=True)

        x = self.conv2(x)
        F.leaky_relu(x, 0.2, inplace=True)

        x = self.validator(x)

        return x


class SimpleFromRgbCombiner(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        return torch.cat([x1, x2], dim=1)


class LinCatFromRgbCombiner(nn.Module):
    def __init__(self, image_channels, bias=False):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=image_channels,
            out_channels=image_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )

    def forward(self, x1, x2):
        x1 = self.conv(x1)
        x1 = F.leaky_relu(x1, 0.2, inplace=True)

        return torch.cat([x1, x2], dim=1)


class CatLinFromRgbCombiner(nn.Module):
    def __init__(self, image_channels, channels, bias=False):
        super().__init__()

        self.in_channels = channels

        self.conv = nn.Conv2d(
            in_channels=channels + image_channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        x = F.leaky_relu(x, 0.2, inplace=True)

        return x

import torch
import torch.nn as nn

import layers as l
import utils


class DiscriminatorFirstBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm, activation_fn, pack=1, bias=False):
        super().__init__()

        self.pack = pack

        self.conv1 = nn.Conv2d(
            in_channels * self.pack,
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

        self.act_fn1 = utils.create_activation_fn(activation_fn, in_channels)
        self.act_fn2 = utils.create_activation_fn(activation_fn, out_channels)

        self.norm1 = utils.create_norm(norm, in_channels)
        self.norm2 = utils.create_norm(norm, out_channels)

        self.avgPool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        if self.pack > 1:
            x = torch.reshape(x, (-1, x.shape[1] * self.pack, x.shape[2], x.shape[3]))

        x = self.norm1(self.act_fn1(self.conv1(x)))
        x = self.norm2(self.act_fn2(self.conv2(x)))

        x = self.avgPool(x)

        return x


class DiscriminatorIntermediateBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm, activation_fn, bias=False):
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

        self.act_fn1 = utils.create_activation_fn(activation_fn, in_channels)
        self.act_fn2 = utils.create_activation_fn(activation_fn, out_channels)

        self.norm1 = utils.create_norm(norm, in_channels)
        self.norm2 = utils.create_norm(norm, out_channels)

        self.avgPool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        x = self.norm1(self.act_fn1(self.conv1(x)))
        x = self.norm2(self.act_fn2(self.conv2(x)))

        x = self.avgPool(x)

        return x


class DiscriminatorLastBlock(nn.Module):
    def __init__(self, in_channels, norm, activation_fn, unary=False, use_mini_batch_std_dev=True, bias=False):
        super().__init__()

        self.unary = unary
        self.useMiniBatchStdDev = use_mini_batch_std_dev

        if self.useMiniBatchStdDev:
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

        self.scorer = nn.Conv2d(
            in_channels,
            2 if self.unary else 1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )

        self.act_fn1 = utils.create_activation_fn(activation_fn, in_channels)
        self.act_fn2 = utils.create_activation_fn(activation_fn, in_channels)

        self.norm1 = utils.create_norm(norm, in_channels)
        self.norm2 = utils.create_norm(norm, in_channels)

    def forward(self, x):
        if self.useMiniBatchStdDev:
            x = self.miniBatchStdDev(x)

        x = self.norm1(self.act_fn1(self.conv1(x)))
        x = self.norm2(self.act_fn2(self.conv2(x)))

        x = self.scorer(x)

        return x.view(-1)


class SimpleFromRgbCombiner(nn.Module):
    def __init__(self, pack: int):
        super().__init__()

        self.pack = pack

    def forward(self, x1, x2):
        if self.pack > 1:
            x1 = torch.reshape(x1, (-1, x1.shape[1] * self.pack, x1.shape[2], x1.shape[3]))

        return torch.cat([x1, x2], dim=1)


class LinCatFromRgbCombiner(nn.Module):
    def __init__(self, image_channels: int, channels: int, pack: int, bias=False):
        super().__init__()

        self.pack = pack

        self.conv = nn.Conv2d(
            in_channels=image_channels * self.pack,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )

    def forward(self, x1, x2):
        if self.pack > 1:
            x1 = torch.reshape(x1, (-1, x1.shape[1] * self.pack, x1.shape[2], x1.shape[3]))

        x1 = self.conv(x1)

        return torch.cat([x1, x2], dim=1)


class CatLinFromRgbCombiner(nn.Module):
    def __init__(self, image_channels: int, channels: int, pack: int, bias=False):
        super().__init__()

        self.pack = pack

        self.conv = nn.Conv2d(
            in_channels=channels + image_channels * self.pack,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )

    def forward(self, x1, x2):
        if self.pack > 1:
            x1 = torch.reshape(x1, (-1, x1.shape[1] * self.pack, x1.shape[2], x1.shape[3]))

        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)

        return x

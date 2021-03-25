import torch
import torch.nn as nn
from torch.nn.utils.spectral_norm import spectral_norm as sn

import layers as l
import utils


class DiscriminatorFirstBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm,
                 activation_fn,
                 image_channels,
                 spectral_norm,
                 msg_skip=True,
                 pack=1,
                 bias=True):
        super().__init__()

        self.msg_skip = msg_skip
        self.pack = pack

        self.conv1 = nn.Conv2d(
            in_channels * self.pack,
            in_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=bias
        )

        self.conv2 = nn.Conv2d(
            in_channels + image_channels * self.pack,
            out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=bias
        )

        self.act_fn1 = utils.create_activation_fn(activation_fn, in_channels)
        self.act_fn2 = utils.create_activation_fn(activation_fn, out_channels)

        self.norm1 = utils.create_norm(norm, in_channels)
        self.norm2 = utils.create_norm(norm, out_channels)

        # elf.down = nn.Sequential(
        #    nn.PixelUnshuffle(2),
        #    nn.Conv2d(
        #        kernel_size=(1, 1),
        #        in_channels=out_channels * 4,
        #        out_channels=out_channels,
        #        padding=(0, 0)
        #    ),
        # )
        self.down = nn.AvgPool2d(2, 2)

        if spectral_norm:
            self.conv1 = sn(self.conv1)
            self.conv2 = sn(self.conv2)

    def forward(self, x, skip=None):
        if self.pack > 1:
            if self.msg_skip:
                skip = torch.reshape(skip, (-1, skip.shape[1] * self.pack, skip.shape[2], skip.shape[3]))

            x = torch.reshape(x, (-1, x.shape[1] * self.pack, x.shape[2], x.shape[3]))

        x = self.norm1(self.act_fn1(self.conv1(x)))
        x = torch.cat([x, skip], dim=1) if self.msg_skip else x
        x = self.norm2(self.act_fn2(self.conv2(x)))

        x = self.down(x)

        return x


class DiscriminatorIntermediateBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm,
                 activation_fn,
                 image_channels,
                 spectral_norm,
                 msg_skip=True,
                 pack=1,
                 bias=True):
        super().__init__()

        self.msg_skip = msg_skip
        self.pack = pack

        # self.conv1 = nn.Conv2d(
        #    in_channels + image_channels * self.pack,
        #    in_channels,
        #    kernel_size=(3, 3),
        #    stride=(1, 1),
        #    padding=(1, 1),
        #    bias=bias
        # )

        self.attention = l.SelfAttention2d(in_channels + image_channels * self.pack, 8, spectral_norm)

        self.conv2 = nn.Conv2d(
            in_channels + image_channels * self.pack,
            out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=bias
        )

        self.act_fn1 = utils.create_activation_fn(activation_fn, in_channels)
        self.act_fn2 = utils.create_activation_fn(activation_fn, out_channels)

        self.norm1 = utils.create_norm(norm, in_channels)
        self.norm2 = utils.create_norm(norm, out_channels)

        # elf.down = nn.Sequential(
        #    nn.PixelUnshuffle(2),
        #    nn.Conv2d(
        #        kernel_size=(1, 1),
        #        in_channels=out_channels * 4,
        #        out_channels=out_channels,
        #        padding=(0, 0)
        #    ),
        # )
        self.down = nn.AvgPool2d(2, 2)

        if spectral_norm:
            # self.conv1 = sn(self.conv1)
            self.conv2 = sn(self.conv2)

    def forward(self, x, skip=None):
        if self.pack > 1:
            if self.msg_skip:
                skip = torch.reshape(skip, (-1, skip.shape[1] * self.pack, skip.shape[2], skip.shape[3]))

            x = torch.cat([x, skip], dim=1)

        # x = self.norm1(self.act_fn1(self.conv1(x)))
        x, _ = self.attention(x)
        # x = torch.cat([x, skip], dim=1) if self.msg_skip else x
        x = self.norm2(self.act_fn2(self.conv2(x)))

        x = self.down(x)

        return x


class DiscriminatorLastBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 norm,
                 activation_fn,
                 image_channels,
                 spectral_norm,
                 msg_skip=True,
                 pack=1,
                 unary=False,
                 use_mini_batch_std_dev=True,
                 bias=True):
        super().__init__()

        self.msg_skip = msg_skip
        self.pack = pack
        self.unary = unary
        self.useMiniBatchStdDev = use_mini_batch_std_dev

        if self.useMiniBatchStdDev:
            self.miniBatchStdDev = l.MinibatchStdDev()

        self.conv1 = nn.Conv2d(
            in_channels + image_channels * self.pack + 1,
            in_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=bias
        )

        self.conv2 = nn.Conv2d(
            in_channels + image_channels * self.pack,
            in_channels,
            kernel_size=(4, 4),
            stride=(1, 1),
            padding=(0, 0),
            bias=bias
        )

        self.scorer = nn.Conv2d(
            in_channels,
            2 if self.unary else 1,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=bias
        )

        self.act_fn1 = utils.create_activation_fn(activation_fn, in_channels)
        self.act_fn2 = utils.create_activation_fn(activation_fn, in_channels)

        self.norm1 = utils.create_norm(norm, in_channels)
        self.norm2 = utils.create_norm(norm, in_channels)

        if spectral_norm:
            self.conv1 = sn(self.conv1)
            self.conv2 = sn(self.conv2)

    def forward(self, x, skip=None):
        if self.pack > 1:
            if self.msg_skip:
                skip = torch.reshape(skip, (-1, skip.shape[1] * self.pack, skip.shape[2], skip.shape[3]))

            x = torch.cat([x, skip], dim=1)

        if self.useMiniBatchStdDev:
            x = self.miniBatchStdDev(x)

        x = self.norm1(self.act_fn1(self.conv1(x)))
        x = torch.cat([x, skip], dim=1) if self.msg_skip else x
        x = self.norm2(self.act_fn2(self.conv2(x)))

        x = self.scorer(x)

        return x.view(-1)

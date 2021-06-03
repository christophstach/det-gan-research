import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm as sn

from layers.adain import AdaptiveInstanceNormalization2d
from layers.pad import EvenPad2d


class UpSkipBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels * scale_factor * scale_factor,
            (1, 1),
            bias=False
        )

        self.shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.shuffle(x)

        return x


class DownSkipBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.unshuffle = nn.PixelUnshuffle(scale_factor)

        self.conv = sn(nn.Conv2d(
            in_channels * scale_factor * scale_factor,
            out_channels,
            (1, 1), bias=False
        ))

    def forward(self, x):
        x = self.unshuffle(x)
        x = self.conv(x)

        return x


class ConvMultiBlock(nn.Module):
    def __init__(self, in_channels, out_channels, spectral_norm=False):
        super().__init__()

        if spectral_norm:
            self.conv2x2 = nn.Sequential(
                EvenPad2d(1),
                sn(nn.Conv2d(in_channels, out_channels // 2, (2, 2)))
            )

            self.conv3x3 = sn(nn.Conv2d(
                in_channels,
                out_channels // 2,
                (3, 3),
                (1, 1),
                (1, 1),
                padding_mode='reflect'
            ))
        else:
            self.conv2x2 = nn.Sequential(
                EvenPad2d(1),
                nn.Conv2d(in_channels, out_channels // 2, (2, 2))
            )

            self.conv3x3 = nn.Conv2d(
                in_channels,
                out_channels // 2,
                (3, 3),
                (1, 1),
                (1, 1),
                padding_mode='reflect'
            )

    def forward(self, x):
        x = torch.cat([
            self.conv2x2(x),
            self.conv3x3(x)
        ], dim=1)

        return x


class UpMultiBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up2x2 = nn.ConvTranspose2d(
            in_channels,
            out_channels // 2,
            (2, 2),
            (2, 2),
            bias=False
        )

        self.up4x4 = nn.ConvTranspose2d(
            in_channels,
            out_channels // 2,
            (4, 4),
            (2, 2),
            (1, 1),
            bias=False
        )

    def forward(self, x):
        x = torch.cat([
            self.up2x2(x),
            self.up4x4(x)
        ], dim=1)

        return x


class DownMultiBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.down2x2 = nn.Sequential(
            EvenPad2d(1),
            sn(nn.Conv2d(in_channels, out_channels // 2, (2, 2), (2, 2)))
        )

        self.down4x4 = nn.Sequential(
            EvenPad2d(1),
            sn(nn.Conv2d(
                in_channels,
                out_channels // 2,
                (4, 4),
                (2, 2),
                (1, 1),
                bias=False
            ))
        )

    def forward(self, x):
        x = torch.cat([
            self.down2x2(x),
            self.down4x4(x)
        ], dim=1)

        return x


class UpResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim, first=False):
        super().__init__()

        self.first = first

        if self.first:
            self.up = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                (4, 4),
                bias=False
            )

            self.skip = UpSkipBlock(in_channels, out_channels, scale_factor=4)
        else:
            self.up = UpMultiBlock(in_channels, out_channels)

            self.adain = AdaptiveInstanceNormalization2d(style_dim, in_channels)

            self.skip = UpSkipBlock(in_channels, out_channels, scale_factor=2)

            # self.up = nn.ConvTranspose2d(
            #    in_channels,
            #    out_channels,
            #    (4, 4),
            #    (2, 2),
            #    (1, 1),
            #    bias=False
            # )

            # self.up = nn.Sequential(
            #    nn.UpsamplingNearest2d(scale_factor=2),
            #    EvenPad2d(1),
            #    nn.Conv2d(in_channels, out_channels, (2, 2))
            # )

            # self.skip = nn.Sequential(
            #    nn.UpsamplingNearest2d(scale_factor=2),
            #    nn.Conv2d(in_channels, out_channels, (1, 1), bias=False)
            # )

        self.conv = ConvMultiBlock(out_channels, out_channels)

        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, w):
        identity1 = self.skip(x)

        if not self.first:
            x = self.adain(x, w)

        x = self.up(x)
        x = self.norm1(x)
        x = F.leaky_relu(x, 0.2)

        identity2 = x

        x = self.conv(x + identity1)
        x = self.norm2(x)
        x = F.leaky_relu(x, 0.2)

        x = x + identity1 + identity2

        return x


class DownResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, last=False):
        super().__init__()

        self.last = last

        if self.last:
            self.down = sn(nn.Conv2d(
                in_channels,
                out_channels,
                (4, 4),
                bias=False
            ))

            self.conv = sn(nn.Conv2d(out_channels, out_channels, (1, 1)))

            # self.skip = nn.Sequential(
            #    nn.AvgPool2d(4, 4),
            #    sn(nn.Conv2d(in_channels, out_channels, (1, 1), bias=False))
            # )

            self.skip = DownSkipBlock(in_channels, out_channels, scale_factor=4)
        else:
            self.down = DownMultiBlock(in_channels, out_channels)

            # self.down = sn(nn.Conv2d(
            #    in_channels,
            #    out_channels,
            #    (4, 4),
            #    (2, 2),
            #    (1, 1),
            #    bias=False,
            #    padding_mode='reflect'
            # ))

            # self.conv = nn.Sequential(
            #    EvenPad2d(1),
            #    sn(nn.Conv2d(out_channels, out_channels, (2, 2)))
            # )

            self.conv = ConvMultiBlock(out_channels, out_channels, spectral_norm=True)

            # self.skip = nn.Sequential(
            #     nn.AvgPool2d(2, 2),
            #    sn(nn.Conv2d(in_channels, out_channels, (1, 1), bias=False))
            # )

            self.skip = DownSkipBlock(in_channels, out_channels, scale_factor=2)

    def forward(self, x):
        identity1 = self.skip(x)

        x = self.down(x)
        x = F.leaky_relu(x, 0.2)

        identity2 = x

        x = self.conv(x + identity1)
        x = F.leaky_relu(x, 0.2)

        x = x + identity1 + identity2

        return x

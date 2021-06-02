import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm as sn

from layers.adain import AdaptiveInstanceNormalization2d
from layers.pad import EvenPad2d


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
            out_channels,
            (2, 2),
            (2, 2),
            bias=False
        )

        self.up3x3 = nn.Sequential(

            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                (3, 3),
                (2, 2),
                (1, 1),
                bias=False
            ),
            EvenPad2d(1)
        )

    def forward(self, x):
        x = torch.cat([
            self.up2x2(x),
            self.up3x3(x)
        ], dim=1)

        return x


class DownMultiBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.down2x2 = nn.Sequential(
            EvenPad2d(1),
            sn(nn.Conv2d(in_channels, out_channels // 2, (2, 2), (2, 2)))
        )

        self.down3x3 = nn.Sequential(
            EvenPad2d(1),
            sn(nn.Conv2d(in_channels, out_channels // 2, (3, 3), (2, 2)))
        )

    def forward(self, x):
        x = torch.cat([
            self.down2x2(x),
            self.down3x3(x)
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
        else:
            self.up = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                (4, 4),
                (2, 2),
                (1, 1),
                bias=False
            )

            # self.up = nn.Sequential(
            #    nn.UpsamplingNearest2d(scale_factor=2),
            #    EvenPad2d(1),
            #    nn.Conv2d(in_channels, out_channels, (2, 2))
            # )

            self.adain = AdaptiveInstanceNormalization2d(style_dim, in_channels)

        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x, w):
        if not self.first:
            x = self.adain(x, w)

        x = self.up(x)
        x = self.norm(x)
        x = F.leaky_relu(x, 0.2)

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

            self.skip = nn.Sequential(
                nn.AvgPool2d(4, 4),
                sn(nn.Conv2d(in_channels, out_channels, (1, 1), bias=False))
            )
        else:
            # self.down = sn(nn.Conv2d(
            #    in_channels,
            #    out_channels,
            #    (4, 4),
            #    (2, 2),
            #    (1, 1),
            #    bias=False,
            #    padding_mode='reflect'
            # ))

            self.down = DownMultiBlock(in_channels, out_channels)

            # self.conv = nn.Sequential(
            #    EvenPad2d(1),
            #    sn(nn.Conv2d(out_channels, out_channels, (2, 2)))
            # )

            self.conv = ConvMultiBlock(out_channels, out_channels, spectral_norm=True)

            self.skip = nn.Sequential(
                nn.AvgPool2d(2, 2),
                sn(nn.Conv2d(in_channels, out_channels, (1, 1), bias=False))
            )

    def forward(self, x):
        identity = self.skip(x)

        x = self.down(x)
        x = F.leaky_relu(x, 0.2)

        x = self.conv(x)
        x = F.leaky_relu(x, 0.2)

        x = x + identity

        return x

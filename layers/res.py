from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm as sn

from layers.adain import AdaptiveInstanceNormalization2d
from layers.pad import EvenPad2d


class UpSkipBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.scale = nn.UpsamplingBilinear2d(scale_factor=scale_factor)
        self.conv = nn.Conv2d(in_channels, out_channels, (1, 1), bias=False)

    def forward(self, x):
        x = self.scale(x)
        x = self.conv(x)

        return x


class DownSkipBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.scale = nn.AvgPool2d(scale_factor, scale_factor)
        self.conv = sn(nn.Conv2d(in_channels, out_channels, (1, 1), bias=False))

    def forward(self, x):
        x = self.scale(x)
        x = self.conv(x)

        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, spectral_norm=False):
        super().__init__()

        assert kernel_size == 2 or kernel_size == 3

        if spectral_norm:
            if kernel_size == 2:
                self.conv = nn.Sequential(
                    EvenPad2d(1),
                    sn(nn.Conv2d(in_channels, out_channels, (2, 2)))
                )
            else:
                self.conv = sn(nn.Conv2d(
                    in_channels,
                    out_channels,
                    (3, 3),
                    (1, 1),
                    (1, 1),
                    padding_mode='reflect'
                ))
        else:
            if kernel_size == 2:
                self.conv = nn.Sequential(
                    EvenPad2d(1),
                    nn.Conv2d(in_channels, out_channels, (2, 2))
                )
            else:
                self.conv = nn.Conv2d(
                    in_channels,
                    out_channels,
                    (3, 3),
                    (1, 1),
                    (1, 1),
                    padding_mode='reflect'
                )

    def forward(self, x):
        x = self.conv(x)

        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            (4, 4),
            (2, 2),
            (1, 1),
            padding_mode='zeros',
            bias=False
        )

    def forward(self, x):
        x = self.conv(x)

        return x


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = sn(nn.Conv2d(
            in_channels,
            out_channels,
            (4, 4),
            (2, 2),
            (1, 1),
            padding_mode='reflect',
            bias=False
        ))

    def forward(self, x):
        x = self.conv(x)

        return x


class UpResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim, first=False, last=False):
        super().__init__()

        self.first = first
        self.last = last

        if self.first:
            self.up = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                (4, 4),
                bias=False
            )
        else:
            self.up = UpBlock(in_channels, out_channels)
            self.adain = AdaptiveInstanceNormalization2d(style_dim, in_channels)

        self.conv = ConvBlock(out_channels, out_channels, 3 if last else 3)

        self.norm1 = nn.BatchNorm2d(out_channels)

        if not self.last:
            self.norm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, w):
        if not self.first:
            x = self.adain(x, w)

        x = self.up(x)
        x = self.norm1(x)
        x = F.leaky_relu(x, 0.2, inplace=True)

        x = self.conv(x)

        if not self.last:
            x = self.norm2(x)
            x = F.leaky_relu(x, 0.2, inplace=True)

        return x


class DownResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, first=False, last=False):
        super().__init__()

        self.last = last

        self.skip = DownSkipBlock(in_channels, out_channels)

        if self.last:
            self.skip = DownSkipBlock(in_channels, out_channels, scale_factor=4)
            self.down = sn(nn.Conv2d(
                in_channels,
                out_channels,
                (4, 4),
                bias=False
            ))

            self.conv = sn(nn.Conv2d(out_channels, out_channels, (1, 1)))
        else:
            self.skip = DownSkipBlock(in_channels, out_channels)
            self.down = DownBlock(in_channels, out_channels)
            self.conv = ConvBlock(out_channels, out_channels, 3 if first else 3, spectral_norm=True)

    def forward(self, x):
        identity = self.skip(x)

        x = self.down(x)
        x = F.leaky_relu(x, 0.2, inplace=True)

        x = self.conv(x)
        x = F.leaky_relu(x, 0.2, inplace=True)

        return x + identity

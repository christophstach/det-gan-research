from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm as sn

from layers.adain import AdaptiveInstanceNormalization2d
from layers.pad import EvenPad2d


class UpResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim, kernel_size, first=False):
        super().__init__()

        if first:
            self.up = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                (4, 4),
                (1, 1),
                (0, 0),
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

        if kernel_size % 2 == 0:
            self.conv = nn.Sequential(
                EvenPad2d(kernel_size),
                nn.Conv2d(out_channels, out_channels, kernel_size)
            )
        else:
            self.conv = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2,
                padding_mode='reflect'
            )

        self.norm1 = AdaptiveInstanceNormalization2d(style_dim, out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, w):
        x = identity = self.up(x)
        x = self.norm1(x, w)
        x = F.leaky_relu(x, 0.2)

        x = self.conv(x)
        x = x + identity
        x = self.norm2(x)
        x = F.leaky_relu(x, 0.2)

        return x


class DownResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, last=False):
        super().__init__()

        if kernel_size % 2 == 0:
            self.conv = nn.Sequential(
                EvenPad2d(kernel_size),
                sn(nn.Conv2d(in_channels, in_channels, kernel_size))
            )

        else:
            self.conv = sn(nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                padding=kernel_size // 2,
                padding_mode='reflect'
            ))

        if last:
            self.down = sn(nn.Conv2d(
                in_channels,
                out_channels,
                (4, 4),
                (1, 1),
                (0, 0),
                bias=False
            ))
        else:
            self.down = sn(nn.Conv2d(
                in_channels,
                out_channels,
                (4, 4),
                (2, 2),
                (1, 1),
                padding_mode='reflect',
                bias=False
            ))

    def forward(self, x):
        identity = x

        x = self.conv(x)
        x = F.leaky_relu(x, 0.2)

        x = x + identity
        x = self.down(x)
        x = F.leaky_relu(x, 0.2)

        return x

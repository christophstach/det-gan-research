import torch.nn as nn
from torch.nn.utils.spectral_norm import spectral_norm as sn

import layers as l
import utils


class GeneratorFirstBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 norm,
                 activation_fn,
                 latent_dimension,
                 spectral_norm,
                 attention,
                 z_skip=True,
                 bias=True):
        super().__init__()

        self.attention = attention
        self.z_skip = z_skip

        if z_skip:
            self.skipper = nn.Conv2d(
                in_channels=latent_dimension,
                out_channels=in_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0)
            )

        if self.attention:
            self.compute1 = l.SelfAttention2d(
                in_channels,
                in_channels,
                spectral_norm=spectral_norm
            )
        else:
            self.compute1 = nn.ConvTranspose2d(
                in_channels,
                in_channels,
                kernel_size=(4, 4),
                stride=(1, 1),
                padding=(0, 0),
                bias=bias
            )

        self.compute2 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=bias
        )

        self.act_fn1 = utils.create_activation_fn(activation_fn, in_channels)
        self.act_fn2 = utils.create_activation_fn(activation_fn, in_channels)

        self.norm1 = utils.create_norm(norm, in_channels)
        self.norm2 = utils.create_norm(norm, in_channels)

        if spectral_norm:
            self.compute1 = sn(self.compute1) if not self.attention else self.compute1
            self.compute2 = sn(self.compute2)

    def forward(self, x, skip=None):
        x = self.norm1(self.act_fn1(self.compute1(x)))
        x = x + self.skipper(skip) if self.z_skip else x
        x = self.norm2(self.act_fn2(self.compute2(x)))

        return x


class GeneratorIntermediateBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm,
                 activation_fn,
                 latent_dimension,
                 spectral_norm,
                 attention,
                 z_skip=True,
                 bias=True):
        super().__init__()

        self.attention = attention
        self.z_skip = z_skip

        if z_skip:
            self.skipper = nn.Conv2d(
                in_channels=latent_dimension,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0)
            )

        if self.attention:
            self.compute1 = l.SelfAttention2d(
                in_channels,
                out_channels,
                spectral_norm=spectral_norm
            )
        else:
            self.compute1 = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=bias
            )

        self.compute2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=bias
        )

        self.act_fn1 = utils.create_activation_fn(activation_fn, out_channels)
        self.act_fn2 = utils.create_activation_fn(activation_fn, out_channels)

        self.norm1 = utils.create_norm(norm, out_channels)
        self.norm2 = utils.create_norm(norm, out_channels)

        self.up = nn.Upsample(
            scale_factor=(2, 2),
            mode='bilinear',
            align_corners=False
        )

        if spectral_norm:
            self.compute1 = sn(self.compute1) if not self.attention else self.compute1
            self.compute2 = sn(self.compute2)

    def forward(self, x, skip=None):
        x = self.up(x)

        x = self.norm1(self.act_fn1(self.compute1(x)))
        x = x + self.skipper(skip) if self.z_skip else x
        x = self.norm2(self.act_fn2(self.compute2(x)))

        return x


class GeneratorLastBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm,
                 activation_fn,
                 latent_dimension,
                 spectral_norm,
                 attention,
                 z_skip=True,
                 bias=False):
        super().__init__()

        self.attention = attention
        self.z_skip = z_skip

        if z_skip:
            self.skipper = nn.Conv2d(
                in_channels=latent_dimension,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0)
            )


        if self.attention:
            self.compute1 = l.SelfAttention2d(
                in_channels,
                out_channels,
                spectral_norm=spectral_norm
            )
        else:
            self.compute1 = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=bias
            )

        self.compute2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=bias
        )

        self.act_fn1 = utils.create_activation_fn(activation_fn, out_channels)
        self.act_fn2 = utils.create_activation_fn(activation_fn, out_channels)

        self.norm1 = utils.create_norm(norm, out_channels)
        self.norm2 = utils.create_norm(norm, out_channels)

        self.up = nn.Upsample(
            scale_factor=(2, 2),
            mode='bilinear',
            align_corners=False
        )

        if spectral_norm:
            self.compute1 = sn(self.compute1) if not self.attention else self.compute1
            self.compute2 = sn(self.compute2)

    def forward(self, x, skip=None):
        x = self.up(x)

        x = self.norm1(self.act_fn1(self.compute1(x)))
        x = x + self.skipper(skip) if self.z_skip else x
        x = self.norm2(self.act_fn2(self.compute2(x)))

        return x

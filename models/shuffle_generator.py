import math

import torch
from torch import nn, Tensor

from layers.adain import AdaptiveInstanceNormalization2d
from layers.blur import StyleBlur
from utils import create_activation_fn, create_norm
from utils.icnr import icnr


# Checkerboard artifact free sub-pixel convolution: A note on sub-pixel convolution,
# resize convolution and convolution resize
# https://arxiv.org/abs/1707.02937

# Checkerboard artifacts free convolutional neural networks
# https://www.cambridge.org/core/journals/apsipa-transactions-on-signal-and-information-processing/article/checkerboard-artifacts-free-convolutional-neural-networks/9F3A72B4581D101881B4A08C09150914#authors-details

class ShuffleGenerator(nn.Module):
    def __init__(self, g_depth, image_size, image_channels, latent_dim):
        super().__init__()

        bias = True
        norm = 'batch'
        activation_fn = 'lrelu'
        disentangler_activation_fn = 'mish'
        padding_mode = 'reflect'

        class FirstBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.conv1 = nn.Conv2d(
                    in_channels,
                    in_channels * 4,
                    (3, 3),
                    (1, 1),
                    (1, 1),
                    padding_mode=padding_mode,
                    bias=bias
                )

                self.act_fn1 = create_activation_fn(activation_fn, in_channels * 4)
                self.norm1 = AdaptiveInstanceNormalization2d(latent_dim, in_channels * 4)

                self.shuffle = nn.PixelShuffle(2)
                self.blur = StyleBlur()

                self.conv2 = nn.Conv2d(
                    in_channels,
                    out_channels,
                    (3, 3),
                    (1, 1),
                    (1, 1),
                    padding_mode=padding_mode,
                    bias=bias
                )

                self.norm2 = create_norm(norm, out_channels)
                self.act_fn2 = create_activation_fn(activation_fn, out_channels)

                self.conv1.weight.data.copy_(icnr(self.conv1.weight, factor=2))

            def forward(self, x, w):
                x = self.conv1(x)
                x = self.norm1(x, w)
                x = self.act_fn1(x)

                x = self.shuffle(x)
                x = self.blur(x)

                x = self.conv2(x)
                x = self.norm2(x)
                x = self.act_fn2(x)

                return x

        class IntermediateBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.conv1 = nn.Conv2d(
                    in_channels,
                    in_channels * 4,
                    (3, 3),
                    (1, 1),
                    (1, 1),
                    padding_mode=padding_mode,
                    bias=bias
                )

                self.norm1 = AdaptiveInstanceNormalization2d(latent_dim, in_channels * 4)
                self.act_fn1 = create_activation_fn(activation_fn, in_channels * 4)

                self.shuffle = nn.PixelShuffle(2)
                self.blur = StyleBlur()

                self.conv2 = nn.Conv2d(
                    in_channels,
                    out_channels,
                    (3, 3),
                    (1, 1),
                    (1, 1),
                    padding_mode=padding_mode,
                    bias=bias
                )

                self.norm2 = create_norm(norm, out_channels)
                self.act_fn2 = create_activation_fn(activation_fn, out_channels)

                self.conv1.weight.data.copy_(icnr(self.conv1.weight, factor=2))

            def forward(self, x, w):
                x = self.conv1(x)
                x = self.norm1(x, w)
                x = self.act_fn1(x)

                x = self.shuffle(x)
                x = self.blur(x)

                x = self.conv2(x)
                x = self.norm2(x)
                x = self.act_fn2(x)

                return x

        class LastBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.conv1 = nn.Conv2d(
                    in_channels,
                    in_channels * 4,
                    (3, 3),
                    (1, 1),
                    (1, 1),
                    padding_mode=padding_mode,
                    bias=bias
                )

                self.norm1 = AdaptiveInstanceNormalization2d(latent_dim, in_channels * 4)
                self.act_fn1 = create_activation_fn(activation_fn, in_channels * 4)

                self.shuffle = nn.PixelShuffle(2)
                self.blur = StyleBlur()

                self.conv2 = nn.Conv2d(
                    in_channels,
                    in_channels,
                    (3, 3),
                    (1, 1),
                    (1, 1),
                    padding_mode=padding_mode,
                    bias=bias
                )

                self.norm2 = create_norm(norm, in_channels)
                self.act_fn2 = create_activation_fn(activation_fn, in_channels)

                self.toRGB = nn.Conv2d(in_channels, out_channels, (1, 1), (1, 1), (0, 0), bias=False)

                self.conv1.weight.data.copy_(icnr(self.conv1.weight, factor=2))

            def forward(self, x, w):
                x = self.conv1(x)
                x = self.norm1(x, w)
                x = self.act_fn1(x)

                x = self.shuffle(x)
                x = self.blur(x)

                x = self.conv2(x)
                x = self.norm2(x)
                x = self.act_fn2(x)

                x = self.toRGB(x)

                return x

        # END block declaration section

        self.channels = [
            *list(reversed([
                2 ** i * g_depth
                for i in range(2, int(math.log2(image_size)))
            ])),
            image_channels
        ]

        self.disentangler = nn.Sequential(
            # nn.Conv2d(latent_dim, latent_dim, (1, 1), (1, 1), (0, 0), bias=bias),
            # create_activation_fn(disentangler_activation_fn, latent_dim)
        )

        self.const = nn.Parameter(
            nn.init.normal_(
                torch.empty(self.channels[0], 4, 4)
            )
        )

        self.blocks = nn.ModuleList()

        for i, channel in enumerate(self.channels):
            if i == 0:  # first
                self.blocks.append(
                    FirstBlock(channel, self.channels[i + 1])
                )
            elif 0 < i < len(self.channels) - 2:  # intermediate
                self.blocks.append(
                    IntermediateBlock(channel, self.channels[i + 1])
                )
            elif i < len(self.channels) - 1:  # last
                self.blocks.append(
                    LastBlock(channel, self.channels[i + 1])
                )

    def forward(self, z: Tensor):
        w = self.disentangler(z.view(z.shape[0], -1, 1, 1))
        x = torch.unsqueeze(self.const, dim=0).repeat(z.shape[0], 1, 1, 1)

        for b in self.blocks:
            x = b(x, w)

        return x, w

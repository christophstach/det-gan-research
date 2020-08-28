import math

import torch

import layers as l
from torch.nn.utils import spectral_norm


class MsgDiscriminator(torch.nn.Module):
    def __init__(self, filter_multiplier, min_filters, max_filters, image_size, image_channels, spectral_normalization) -> None:
        super().__init__()

        self.blocks = torch.nn.ModuleList()
        self.from_rgb_combiners = torch.nn.ModuleList()

        discriminator_filters = [
            2 ** (x + 1) * filter_multiplier
            for x in range(1, int(math.log2(image_size)))
        ]

        if min_filters > 0:
            discriminator_filters = [
                filter if filter > min_filters
                else min_filters
                for filter in discriminator_filters
            ]

        if max_filters > 0:
            discriminator_filters = [
                filter if filter < max_filters
                else max_filters
                for filter in discriminator_filters
            ]

        for i, _ in enumerate(discriminator_filters):
            if i == 0:
                self.blocks.append(
                    l.MsgDiscriminatorFirstBlock(
                        image_channels,
                        discriminator_filters[i + 1]
                    )
                )
            elif i < len(discriminator_filters) - 1:
                self.blocks.append(
                    l.MsgDiscriminatorIntermediateBlock(
                        discriminator_filters[i] + image_channels,
                        discriminator_filters[i + 1]
                    )
                )
            else:
                self.blocks.append(
                    l.MsgDiscriminatorLastBlock(
                        discriminator_filters[i] + image_channels,
                        1
                    )
                )

            self.from_rgb_combiners.append(
                l.SimpleFromRgbCombiner()
            )

        if spectral_normalization:
            for block in self.blocks:
                block.conv1 = spectral_norm(block.conv1)
                block.conv2 = spectral_norm(block.conv2)

    def forward(self, x) -> torch.Tensor:
        x = list(reversed(x))
        x_forward = self.blocks[0](x[0])

        for data, block, from_rgb in zip(x[1:], self.blocks[1:], self.from_rgb_combiners):
            x_forward = from_rgb(data, x_forward)
            x_forward = block(x_forward)

        return x_forward

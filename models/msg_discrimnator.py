import math

import torch
from torch.nn.utils import spectral_norm

import layers as l


class MsgDiscriminator(torch.nn.Module):
    def __init__(self,
                 filter_multiplier: int,
                 min_filters: int,
                 max_filters: int,
                 image_size: int,
                 image_channels: int,
                 spectral_normalization: bool) -> None:

        super().__init__()

        self.blocks = torch.nn.ModuleList()
        self.from_rgb_combiners = torch.nn.ModuleList()

        discriminator_filters = [
            2 ** (x + 1) * filter_multiplier
            for x in range(1, int(math.log2(image_size)))
        ]

        if min_filters > 0:
            discriminator_filters = [
                d_filter if d_filter > min_filters
                else min_filters
                for d_filter in discriminator_filters
            ]

        if max_filters > 0:
            discriminator_filters = [
                d_filter if d_filter < max_filters
                else max_filters
                for d_filter in discriminator_filters
            ]

        for i, _ in enumerate(discriminator_filters):
            if i == 0:
                self.blocks.append(
                    l.MsgDiscriminatorFirstBlock(
                        image_channels,
                        discriminator_filters[i + 1]
                    )
                )

                self.from_rgb_combiners.append(
                    l.LinCatFromRgbCombiner(image_channels=image_channels, channels=discriminator_filters[i + 1])
                )
            elif i < len(discriminator_filters) - 1:
                self.blocks.append(
                    l.MsgDiscriminatorIntermediateBlock(
                        discriminator_filters[i] * 2,
                        discriminator_filters[i + 1]
                    )
                )

                self.from_rgb_combiners.append(
                    l.LinCatFromRgbCombiner(image_channels=image_channels, channels=discriminator_filters[i + 1])
                )
            else:
                self.blocks.append(
                    l.MsgDiscriminatorLastBlock(
                        discriminator_filters[i] * 2,
                        1
                    )
                )

        if spectral_normalization:
            for block in self.blocks:
                block.conv1 = spectral_norm(block.conv1)
                block.conv2 = spectral_norm(block.conv2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = list(reversed(x))
        x_forward = self.blocks[0](x[0])

        for data, block, from_rgb in zip(x[1:], self.blocks[1:], self.from_rgb_combiners):
            x_forward = from_rgb(data, x_forward)
            x_forward = block(x_forward)

        return x_forward

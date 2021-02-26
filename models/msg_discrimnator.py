import math

import torch
from torch.nn.utils import spectral_norm

import layers as l

from typing import List


class MsgDiscriminator(torch.nn.Module):
    def __init__(self,
                 depth: int,
                 min_filters: int,
                 max_filters: int,
                 image_size: int,
                 image_channels: int,
                 normalization: str,
                 activation_fn: str,
                 spectral_normalization: bool,
                 msg: bool,
                 unary: bool,
                 pack: int) -> None:

        super().__init__()

        self.msg = msg
        self.pack = pack
        self.blocks = torch.nn.ModuleList()

        if self.msg:
            self.from_rgb_combiners = torch.nn.ModuleList()

        discriminator_filters = [
            2 ** (x + 1) * depth
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
            simple_from_rgb_combiner = False

            if self.msg:
                additional_filters = 3 if simple_from_rgb_combiner else discriminator_filters[i]
            else:
                additional_filters = 0

            if i == 0:
                self.blocks.append(
                    l.MsgDiscriminatorFirstBlock(
                        image_channels,
                        discriminator_filters[i + 1],
                        normalization,
                        activation_fn,
                        pack=pack
                    )
                )

                if self.msg:
                    self.from_rgb_combiners.append(
                        l.LinCatFromRgbCombiner(
                            image_channels=image_channels,
                            channels=discriminator_filters[i + 1],
                            pack=self.pack
                        )
                    )
            elif i < len(discriminator_filters) - 1:
                self.blocks.append(
                    l.MsgDiscriminatorIntermediateBlock(
                        discriminator_filters[i] + additional_filters,
                        discriminator_filters[i + 1],
                        normalization,
                        activation_fn
                    )
                )

                if self.msg:
                    self.from_rgb_combiners.append(
                        l.LinCatFromRgbCombiner(
                            image_channels=image_channels,
                            channels=discriminator_filters[i + 1],
                            pack=self.pack
                        )
                    )
            else:
                self.blocks.append(
                    l.MsgDiscriminatorLastBlock(
                        discriminator_filters[i] + additional_filters,
                        normalization,
                        activation_fn,
                        unary=unary
                    )
                )

        if spectral_normalization:
            for block in self.blocks:
                block.conv1 = spectral_norm(block.conv1)
                block.conv2 = spectral_norm(block.conv2)

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        if self.msg:
            x = list(reversed(x))
            x_forward = self.blocks[0](x[0])

            for data, block, from_rgb in zip(x[1:], self.blocks[1:], self.from_rgb_combiners):
                x_forward = from_rgb(data, x_forward)
                x_forward = block(x_forward)
        else:
            x_forward = x[0]

            for block in self.blocks:
                x_forward = block(x_forward)

        return x_forward

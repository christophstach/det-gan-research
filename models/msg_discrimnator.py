import math
from typing import List

import torch
from torch import Tensor
from torch.nn.utils import spectral_norm

import layers as l


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
            if i == 0:
                self.blocks.append(
                    l.DiscriminatorFirstBlock(
                        image_channels,
                        discriminator_filters[i + 1],
                        normalization,
                        activation_fn,
                        pack=self.pack
                    )
                )
            elif i < len(discriminator_filters) - 1:
                self.blocks.append(
                    l.DiscriminatorIntermediateBlock(
                        discriminator_filters[i],
                        discriminator_filters[i + 1],
                        normalization,
                        activation_fn,
                        image_channels,
                        msg_skip=self.msg,
                        pack=self.pack
                    )
                )
            else:
                self.blocks.append(
                    l.DiscriminatorLastBlock(
                        discriminator_filters[i],
                        normalization,
                        activation_fn,
                        image_channels,
                        msg_skip=self.msg,
                        pack=self.pack,
                        unary=unary
                    )
                )

        if spectral_normalization:
            for block in self.blocks:
                block.conv1 = spectral_norm(block.conv1)
                block.conv2 = spectral_norm(block.conv2)

    def forward(self, x: List[Tensor]) -> Tensor:
        if self.msg:
            x = list(reversed(x))
            x_forward = self.blocks[0](x[0])

            for skip, block in zip(x[1:], self.blocks[1:]):
                x_forward = block(x_forward, skip)
        else:
            x_forward = x[0]

            for block in self.blocks:
                x_forward = block(x_forward)

        return x_forward

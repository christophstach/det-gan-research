from typing import List, Tuple

import math
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

import layers as l


class MsgGenerator(nn.Module):
    def __init__(self,
                 depth: int,
                 min_filters: int,
                 max_filters: int,
                 image_size: int,
                 image_channels: int,
                 latent_dimension: int,
                 spectral_normalization: bool) -> None:

        super().__init__()

        # self.w_network = nn.Sequential(
        #    nn.Conv2d(latent_dimension, latent_dimension, kernel_size=1)
        # )

        self.blocks = torch.nn.ModuleList()
        self.to_rgb_converters = torch.nn.ModuleList()

        generator_filters = [
            2 ** (x + 1) * depth
            for x in reversed(range(1, int(math.log2(image_size))))
        ]

        if min_filters > 0:
            generator_filters = [
                g_filter if g_filter > min_filters
                else min_filters
                for g_filter in generator_filters
            ]

        if max_filters > 0:
            generator_filters = [
                g_filter if g_filter < max_filters
                else max_filters
                for g_filter in generator_filters
            ]

        generator_filters[0] = latent_dimension

        for i, _ in enumerate(generator_filters):
            if i == 0:
                self.blocks.append(
                    l.MsgGeneratorFirstBlock(
                        generator_filters[i]
                    )
                )
            elif i < len(generator_filters) - 1:
                self.blocks.append(
                    l.MsgGeneratorIntermediateBlock(
                        generator_filters[i - 1],
                        generator_filters[i],
                    )
                )
            else:
                self.blocks.append(
                    l.MsgGeneratorLastBlock(
                        generator_filters[i - 1],
                        generator_filters[i]
                    )
                )

            self.to_rgb_converters.append(
                torch.nn.Conv2d(
                    in_channels=generator_filters[i],
                    out_channels=image_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )
            )

        if spectral_normalization:
            for block in self.blocks:
                block.conv1 = spectral_norm(block.conv1)
                block.conv2 = spectral_norm(block.conv2)

    def forward(self, z: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        outputs = []

        z = z.view(z.shape[0], -1, 1, 1)
        # w = self.w_network(z)
        w = z
        x = w

        for block, to_rgb in zip(self.blocks, self.to_rgb_converters):
            x = block(x)
            output = torch.tanh(to_rgb(x))
            outputs.append(output)

        return outputs, w

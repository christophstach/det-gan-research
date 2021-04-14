from torch import nn
from torch.nn.functional import interpolate
from torch.nn.utils import spectral_norm

from utils import create_activation_fn


class ResNetDiscriminator(nn.Module):
    def __init__(self, d_depth: int, image_channels: int, score_dimension: int = 1) -> None:
        super().__init__()



        class FirstBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.down = nn.Sequential(
                    Downscale(),
                    spectral_norm(nn.Conv2d(in_channels, out_channels, (1, 1), (1, 1), (0, 0))),
                )

                self.compute = nn.Sequential(
                    spectral_norm(nn.Conv2d(in_channels, out_channels, (4, 4), (2, 2), (1, 1))),
                    create_activation_fn('lrelu', out_channels),
                )

            def forward(self, x):
                identity = self.down(x)
                x = self.compute(x)

                return x + identity

        class IntermediateBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.down = nn.Sequential(
                    Downscale(),
                    spectral_norm(nn.Conv2d(in_channels, out_channels, (1, 1), (1, 1), (0, 0))),
                )

                self.compute = nn.Sequential(
                    spectral_norm(nn.Conv2d(in_channels, out_channels, (4, 4), (2, 2), (1, 1))),
                    create_activation_fn('lrelu', out_channels),
                )

            def forward(self, x):
                identity = self.down(x)
                x = self.compute(x)

                return x + identity

        class LastBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.compute = spectral_norm(nn.Conv2d(in_channels, out_channels, (4, 4), (1, 1), (0, 0)))

            def forward(self, x):
                x = self.compute(x)

                return x

        # END block declaration section

        self.channels = [
            image_channels,
            4 * d_depth,
            8 * d_depth,
            16 * d_depth,
            32 * d_depth,
            64 * d_depth,
            # 128 * d_depth,
            score_dimension
        ]

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

    def forward(self, x):
        for b in self.blocks:
            x = b(x)

        return x

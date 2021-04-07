from torch import nn
from torch.nn.functional import interpolate
from torch.nn.utils import spectral_norm


class ResNetDiscriminator(nn.Module):
    def __init__(self, d_depth: int, image_channels: int, score_dimension: int = 1) -> None:
        super().__init__()

        class Downscale(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return interpolate(
                    input=x,
                    scale_factor=(0.5, 0.5),
                    mode='bilinear',
                    align_corners=False,
                    recompute_scale_factor=False
                )

        class FirstBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.down = nn.Sequential(
                    Downscale(),
                    spectral_norm(nn.Conv2d(in_channels, out_channels, (1, 1), (1, 1), (0, 0))),
                )

                # self.compute = spectral_norm(nn.Conv2d(in_channels, out_channels, (4, 4), (2, 2), (1, 1)))

                self.compute = nn.Sequential(
                    nn.PixelUnshuffle(2),
                    spectral_norm(nn.Conv2d(in_channels * 4, out_channels, (1, 1), (1, 1), (0, 0)))
                )

                self.activation = nn.LeakyReLU(0.2)

            def forward(self, x):
                identity = self.down(x)
                x = self.activation(self.compute(x))

                return x + identity

        class IntermediateBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.down = nn.Sequential(
                    Downscale(),
                    spectral_norm(nn.Conv2d(in_channels, out_channels, (1, 1), (1, 1), (0, 0))),
                )

                # self.compute = spectral_norm(nn.Conv2d(in_channels, out_channels, (4, 4), (2, 2), (1, 1)))

                self.compute = nn.Sequential(
                    nn.PixelUnshuffle(2),
                    spectral_norm(nn.Conv2d(in_channels * 4, out_channels, (1, 1), (1, 1), (0, 0)))
                )

                self.activation = nn.LeakyReLU(0.2)

            def forward(self, x):
                identity = self.down(x)
                x = self.activation(self.compute(x))

                return x + identity

        class LastBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.compute = spectral_norm(nn.Conv2d(in_channels, out_channels, (4, 4), (1, 1), (0, 0)))

            def forward(self, x):
                x = self.compute(x)

                return x

        # END block declaration section

        channels = [
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

        for i, channel in enumerate(channels):
            if i == 0:  # first
                self.blocks.append(
                    FirstBlock(channel, channels[i + 1])
                )
            elif 0 < i < len(channels) - 2:  # intermediate
                self.blocks.append(
                    IntermediateBlock(channel, channels[i + 1])
                )
            elif i < len(channels) - 1:  # last
                self.blocks.append(
                    LastBlock(channel, channels[i + 1])
                )

    def forward(self, x):
        for b in self.blocks:
            x = b(x)

        return x

from torch import nn
from torch.nn.functional import interpolate

from utils import create_activation_fn


class ResNetGenerator(nn.Module):
    def __init__(self, g_depth: int, image_channels: int, latent_dimension: int) -> None:
        super().__init__()

        class Upscale(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return interpolate(
                    input=x,
                    scale_factor=(2, 2),
                    mode='bilinear',
                    align_corners=False
                )

        class FirstBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.compute = nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, (4, 4), (1, 1), (0, 0)),
                    create_activation_fn('lrelu', out_channels),
                )

            def forward(self, x):
                x = self.compute(x)

                return x

        class IntermediateBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.up = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, (1, 1), (1, 1), (0, 0)),
                    Upscale()
                )

                self.compute = nn.Sequential(
                    Upscale(),
                    nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1)),
                    create_activation_fn('lrelu', out_channels),
                )

            def forward(self, x):
                identity = self.up(x)
                x = self.compute(x)

                return x + identity

        class LastBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.up = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, (1, 1), (1, 1), (0, 0)),
                    Upscale()
                )

                self.compute = nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, (4, 4), (2, 2), (1, 1)),
                    create_activation_fn('tanh', out_channels),
                )

            def forward(self, x):
                identity = self.up(x)
                x = self.compute(x)

                return 0.5 * x + 0.5 * identity

        # END block declaration section

        channels = [
            latent_dimension,
            # 128 * g_depth,
            64 * g_depth,
            32 * g_depth,
            16 * g_depth,
            8 * g_depth,
            4 * g_depth,
            image_channels
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

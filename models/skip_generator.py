from torch import Tensor, sigmoid, randn_like
from torch import nn

from utils import create_norm, create_activation_fn, create_upscale


class SkipGenerator(nn.Module):
    def __init__(self, g_depth: int, image_channels: int, latent_dimension: int) -> None:
        super().__init__()

        class FirstBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.compute1 = nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, (4, 4), (1, 1), (0, 0)),
                    create_activation_fn('lrelu', out_channels),
                    create_norm('pixel', out_channels)
                )

                self.compute2 = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), (1, 1)),
                    create_activation_fn('lrelu', out_channels),
                    create_norm('pixel', out_channels)
                )

                self.toRGB = nn.Sequential(
                    nn.Conv2d(out_channels, image_channels, (1, 1), (1, 1), (0, 0)),
                    nn.Tanh()
                )

                self.up = create_upscale('interpolate')
                self.alpha = nn.Parameter(Tensor(1).fill_(1.0))

            def forward(self, x, identity):
                x = self.compute1(x)
                x = self.compute2(x + randn_like(x))
                rgb = self.toRGB(x)
                rgb = self.up(rgb)

                return x, rgb

        class IntermediateBlock(nn.Module):
            def __init__(self, in_channels, out_channels, size=1):
                super().__init__()

                self.compute1 = nn.Sequential(
                    create_upscale('interpolate'),
                    nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1)),
                    create_activation_fn('lrelu', out_channels),
                    create_norm('pixel', out_channels)
                )

                self.compute2 = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), (1, 1)),
                    create_activation_fn('lrelu', out_channels),
                    create_norm('pixel', out_channels)
                )

                self.toRGB = nn.Sequential(
                    nn.Conv2d(out_channels, image_channels, (1, 1), (1, 1), (0, 0)),
                    nn.Tanh()
                )

                self.up = create_upscale('interpolate')
                self.alpha = nn.Parameter(Tensor(1).fill_(1.0))
                self.beta = nn.Parameter(Tensor(image_channels, size, size).fill_(0.0))

            def forward(self, x, rgb, identity):
                x = self.compute1(x)
                x = self.compute2(x + randn_like(x))
                percentage = sigmoid(self.beta)
                rgb = percentage * self.toRGB(x) * (1.0 - percentage) + rgb
                rgb = self.up(rgb)

                return x, rgb

        class LastBlock(nn.Module):
            def __init__(self, in_channels, out_channels, size=1):
                super().__init__()

                self.compute1 = nn.Sequential(
                    create_upscale('interpolate'),
                    nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1)),
                    create_activation_fn('lrelu', out_channels),
                    create_norm('pixel', out_channels)
                )

                self.compute2 = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), (1, 1)),
                    create_activation_fn('lrelu', out_channels),
                    create_norm('pixel', out_channels)
                )

                self.toRGB = nn.Sequential(
                    nn.Conv2d(out_channels, image_channels, (1, 1), (1, 1), (0, 0)),
                    nn.Tanh()
                )

                self.alpha = nn.Parameter(Tensor(1).fill_(1.0))
                self.beta = nn.Parameter(Tensor(image_channels, size, size).fill_(0.0))

            def forward(self, x, rgb, identity):
                x = self.compute1(x)
                x = self.compute2(x + randn_like(x))
                percentage = sigmoid(self.beta)
                rgb = percentage * self.toRGB(x) + (1.0 - percentage) * rgb

                return rgb

        # END block declaration section

        self.channels = [
            latent_dimension,
            64 * g_depth,
            32 * g_depth,
            16 * g_depth,
            8 * g_depth,
            4 * g_depth,
            image_channels
        ]

        self.blocks = nn.ModuleList()

        for i, channel in enumerate(self.channels):
            if i == 0:  # first
                self.blocks.append(
                    FirstBlock(channel, self.channels[i + 1])
                )
            elif 0 < i < len(self.channels) - 2:  # intermediate
                self.blocks.append(
                    IntermediateBlock(channel, self.channels[i + 1], 4 * 2 ** i)
                )
            elif i < len(self.channels) - 1:  # last
                self.blocks.append(
                    LastBlock(channel, self.channels[i + 1], 4 * 2 ** i)
                )

    def forward(self, x):
        identity = x
        rgb = None

        for i, b in enumerate(self.blocks):
            if i == 0:  # first
                x, rgb = b(x, identity)
            elif 0 < i < len(self.channels) - 2:  # intermediate
                x, rgb = b(x, rgb, identity)
            elif i < len(self.channels) - 1:  # last
                x = b(x, rgb, identity)

        return x

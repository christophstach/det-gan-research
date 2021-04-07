from torch import nn

from torch.nn.functional import interpolate

import layers as l


class SkipGenerator(nn.Module):
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

                self.compute = nn.ConvTranspose2d(in_channels, out_channels, (4, 4), (1, 1), (0, 0))

                self.activation = nn.LeakyReLU(0.2)

                self.norm = l.PixelNorm()

                self.toRGB = nn.Sequential(
                    nn.Conv2d(out_channels, image_channels, (1, 1), (1, 1), (0, 0)),
                    nn.Tanh()
                )

                self.up = Upscale()

            def forward(self, x):
                x = self.norm(self.activation(self.compute(x)))
                identity = self.toRGB(x)
                identity = self.up(identity)

                return x, identity

        class IntermediateBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                # self.compute = nn.ConvTranspose2d(in_channels, out_channels, (4, 4), (2, 2), (1, 1))

                # self.compute = nn.Sequential(
                #    Upscale(),
                #    nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1))
                # )

                self.compute = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels * 4, (1, 1), (1, 1), (0, 0)),
                    nn.PixelShuffle(2)
                )

                self.activation = nn.LeakyReLU(0.2)

                self.norm = l.PixelNorm()

                self.toRGB = nn.Sequential(
                    nn.Conv2d(out_channels, image_channels, (1, 1), (1, 1), (0, 0)),
                    nn.Tanh()
                )

                self.up = Upscale()

            def forward(self, x, identity):
                x = self.norm(self.activation(self.compute(x)))
                identity = self.toRGB(x) + identity
                identity = self.up(identity)

                return x, identity

        class LastBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                # self.compute = nn.ConvTranspose2d(in_channels, out_channels, (4, 4), (2, 2), (1, 1))

                # self.compute = nn.Sequential(
                #    Upscale(),
                #    nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1))
                # )

                self.compute = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels * 4, (1, 1), (1, 1), (0, 0)),
                    nn.PixelShuffle(2)
                )

                self.activation = nn.LeakyReLU(0.2)

                self.norm = l.PixelNorm()

                self.toRGB = nn.Sequential(
                    nn.Conv2d(out_channels, image_channels, (1, 1), (1, 1), (0, 0)),
                    nn.Tanh()
                )

            def forward(self, x, identity):
                x = self.norm(self.activation(self.compute(x)))
                identity = self.toRGB(x) + identity

                return identity

        # END block declaration section

        channels = [
            latent_dimension,
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
        identity = None

        for i, b in enumerate(self.blocks):
            if i == 0:  # first
                x, identity = b(x)
            elif 0 < i < len(self.blocks) - 2:  # intermediate
                x, identity = b(x, identity)
            elif i < len(self.blocks) - 1:  # last
                x, identity = b(x, identity)

        return identity

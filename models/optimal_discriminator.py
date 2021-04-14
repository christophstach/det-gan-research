import torch
from torch import nn
from torch.nn.utils import spectral_norm

from utils import create_activation_fn, create_downscale


class OptimalDiscriminator(nn.Module):
    def __init__(self, d_depth: int, image_channels: int, score_dimension: int = 1) -> None:
        super().__init__()

        def weights_init(m):
            class_name = m.__class__.__name__

            if class_name.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif class_name.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

        class FirstBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.down = nn.Sequential(
                    create_downscale('avgpool'),
                    spectral_norm(nn.Conv2d(in_channels, out_channels, (1, 1), (1, 1), (0, 0)))
                )

                self.compute1 = nn.Sequential(
                    spectral_norm(nn.Conv2d(in_channels, in_channels, (3, 3), (1, 1), (1, 1))),
                    create_activation_fn('lrelu', out_channels),
                )

                self.compute2 = nn.Sequential(
                    spectral_norm(nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1))),
                    create_activation_fn('lrelu', out_channels),
                    create_downscale('avgpool'),
                )

            def forward(self, x):
                identity = self.down(x)
                x = self.compute1(x)
                x = self.compute2(x)

                return x + identity

        class IntermediateBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.down = nn.Sequential(
                    create_downscale('avgpool'),
                    spectral_norm(nn.Conv2d(in_channels, out_channels, (1, 1), (1, 1), (0, 0)))
                )

                self.compute1 = nn.Sequential(
                    spectral_norm(nn.Conv2d(in_channels, in_channels, (3, 3), (1, 1), (1, 1))),
                    create_activation_fn('lrelu', out_channels),
                )

                self.compute2 = nn.Sequential(
                    spectral_norm(nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1))),
                    create_activation_fn('lrelu', out_channels),
                    create_downscale('avgpool'),
                )

            def forward(self, x):
                identity = self.down(x)
                x = self.compute1(x)
                x = self.compute2(x)

                return x + identity

        class LastBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.down = spectral_norm(nn.Conv2d(in_channels, out_channels, (4, 4), (1, 1), (0, 0)))

                self.compute1 = nn.Sequential(
                    spectral_norm(nn.Conv2d(in_channels, in_channels, (3, 3), (1, 1), (1, 1))),
                    create_activation_fn('lrelu', out_channels),
                )

                self.compute2 = nn.Sequential(
                    spectral_norm(nn.Conv2d(in_channels, out_channels, (4, 4), (1, 1), (0, 0))),
                    create_activation_fn('lrelu', out_channels),
                )

                self.scorer1 = nn.Sequential(
                    spectral_norm(nn.Conv2d(out_channels, out_channels, (1, 1), (1, 1), (0, 0))),
                    create_activation_fn('lrelu', out_channels),
                )
                self.scorer2 = spectral_norm(nn.Conv2d(out_channels, out_channels, (1, 1), (1, 1), (0, 0)))

            def forward(self, x):
                identity = self.down(x)
                x = self.compute1(x)
                x = self.compute2(x)
                x = self.scorer1(x + identity)
                x = self.scorer2(x)

                return x

        # END block declaration section

        channels = [
            image_channels,
            4 * d_depth,  # 8
            8 * d_depth,  # 16
            16 * d_depth,  # 32
            32 * d_depth,  # 64
            64 * d_depth,  # 128
            # 128 * d_depth,  # 256
            score_dimension
        ]

        self.blocks = nn.ModuleList()
        self.pack = 2

        for i, channel in enumerate(channels):
            if i == 0:  # first
                self.blocks.append(
                    FirstBlock(channel * self.pack, channels[i + 1])
                )
            elif 0 < i < len(channels) - 2:  # intermediate
                self.blocks.append(
                    IntermediateBlock(channel, channels[i + 1])
                )
            elif i < len(channels) - 1:  # last
                self.blocks.append(
                    LastBlock(channel, channels[i + 1])
                )

        self.apply(weights_init)

    def forward(self, x):
        x = torch.reshape(x, (-1, x.shape[1] * self.pack, x.shape[2], x.shape[3]))

        for b in self.blocks:
            x = b(x)

        return x

from torch import nn, Tensor
from torch.nn.functional import interpolate

from utils import create_norm, create_activation_fn


class OptimalGenerator(nn.Module):
    def __init__(self, g_depth: int, image_channels: int, latent_dimension: int) -> None:
        super().__init__()

        def weights_init(m):
            class_name = m.__class__.__name__

            if class_name.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif class_name.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

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

                self.map = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, (1, 1), (1, 1), (0, 0)),
                    create_activation_fn('lrelu', in_channels),
                    nn.Conv2d(in_channels, in_channels, (1, 1), (1, 1), (0, 0)),
                    create_activation_fn('lrelu', in_channels),
                    nn.Conv2d(in_channels, in_channels, (1, 1), (1, 1), (0, 0)),
                    create_activation_fn('lrelu', in_channels),
                    nn.Conv2d(in_channels, in_channels, (1, 1), (1, 1), (0, 0)),
                    create_activation_fn('lrelu', in_channels),
                    nn.Conv2d(in_channels, in_channels, (1, 1), (1, 1), (0, 0)),
                    create_activation_fn('lrelu', in_channels),
                    nn.Conv2d(in_channels, in_channels, (1, 1), (1, 1), (0, 0)),
                    create_activation_fn('lrelu', in_channels),
                    nn.Conv2d(in_channels, in_channels, (1, 1), (1, 1), (0, 0)),
                    create_activation_fn('lrelu', in_channels),
                    nn.Conv2d(in_channels, in_channels, (1, 1), (1, 1), (0, 0)),
                    create_activation_fn('lrelu', in_channels)
                )

                self.up = nn.Conv2d(latent_dimension, out_channels, (1, 1), (1, 1))

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

            def forward(self, x):
                # x = self.map(x)
                identity = x
                x = self.compute1(x) + self.up(identity)
                x = self.compute2(x)

                return x, identity

        class IntermediateBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.up = nn.Conv2d(latent_dimension, out_channels, (1, 1), (1, 1))

                self.compute1 = nn.Sequential(
                    Upscale(),
                    nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1)),
                    create_activation_fn('lrelu', out_channels),
                    create_norm('pixel', out_channels)
                )

                self.compute2 = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), (1, 1)),
                    create_activation_fn('lrelu', out_channels),
                    create_norm('pixel', out_channels)
                )

            def forward(self, x, identity):
                x = self.compute1(x) + self.up(identity)
                x = self.compute2(x)

                return x

        class LastBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.up = nn.Conv2d(latent_dimension, out_channels, (1, 1), (1, 1))

                self.compute1 = nn.Sequential(
                    Upscale(),
                    nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1)),
                    create_activation_fn('lrelu', out_channels),
                    create_norm('pixel', out_channels)
                )

                self.compute2 = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), (1, 1)),
                    create_activation_fn('lrelu', out_channels),
                    create_norm('pixel', out_channels)
                )

                self.noise1 = nn.Parameter(Tensor(out_channels).fill_(1.0))
                self.noise2 = nn.Parameter(Tensor(out_channels).fill_(1.0))

                self.toRGB = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, (1, 1), (1, 1), (0, 0)),
                    nn.Tanh()
                )




            def forward(self, x, identity):
                x = self.compute1(x) + self.up(identity)
                x = self.compute2(x)
                x = self.toRGB(x)

                return x

        # END block declaration section

        self.channels = [
            latent_dimension,
            # 128 * g_depth,  # 256
            64 * g_depth,  # 128
            32 * g_depth,  # 64
            16 * g_depth,  # 32
            8 * g_depth,  # 16
            4 * g_depth,  # 8
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
                    IntermediateBlock(channel, self.channels[i + 1])
                )
            elif i < len(self.channels) - 1:  # last
                self.blocks.append(
                    LastBlock(channel, self.channels[i + 1])
                )

        self.apply(weights_init)

    def forward(self, x):
        identity = None

        for i, b in enumerate(self.blocks):
            if i == 0:  # first
                x, identity = b(x)
            elif 0 < i < len(self.channels) - 2:  # intermediate
                x = b(x, identity)
            elif i < len(self.channels) - 1:  # last
                x = b(x, identity)

        return x

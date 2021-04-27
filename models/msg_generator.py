import math

from torch import nn

from layers.eql import EqlConv2d
from utils import create_activation_fn, create_norm, create_upscale
from torch.nn.utils import spectral_norm as sn


class MsgGenerator(nn.Module):
    def __init__(self, g_depth, image_size, image_channels, latent_dim):
        super().__init__()

        norm = 'passthrough'
        activation_fn = 'lrelu'
        upscale = 'nearest'
        eql = False

        class Conv(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                if eql:
                    self.conv = sn(
                        EqlConv2d(
                            in_channels,
                            out_channels,
                            (3, 3),
                            (1, 1),
                            (1, 1),
                            padding_mode='replicate'
                        )
                    )

                else:
                    self.conv = sn(
                        nn.Conv2d(
                            in_channels,
                            out_channels,
                            (3, 3),
                            (1, 1),
                            (1, 1),
                            padding_mode='replicate'
                        )
                    )

            def forward(self, x):
                return self.conv(x)

        class FirstBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.compute1 = Conv(in_channels, out_channels)
                self.compute2 = Conv(out_channels, out_channels)

                self.norm0 = create_norm(norm, in_channels)
                self.norm1 = create_norm(norm, out_channels)
                self.norm2 = create_norm(norm, out_channels)

                self.act_fn1 = create_activation_fn(activation_fn, out_channels)
                self.act_fn2 = create_activation_fn(activation_fn, out_channels)

                if eql:
                    self.toRGB = nn.Sequential(
                        EqlConv2d(out_channels, image_channels, (1, 1), (1, 1), (0, 0)),
                        nn.Tanh()
                    )
                else:
                    self.toRGB = nn.Sequential(
                        nn.Conv2d(out_channels, image_channels, (1, 1), (1, 1), (0, 0)),
                        nn.Tanh()
                    )

            def forward(self, x):
                x = self.norm0(x)

                x = self.compute1(x)
                x = self.norm1(x)
                x = self.act_fn1(x)

                x = self.compute2(x)
                x = self.norm2(x)
                x = self.act_fn2(x)

                return x, self.toRGB(x)

        class IntermediateBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.compute1 = nn.Sequential(
                    create_upscale(upscale),
                    Conv(in_channels, out_channels),
                )
                # self.compute1 = create_upscale(upscale, in_channels, out_channels)
                self.compute2 = Conv(out_channels, out_channels)

                self.norm1 = create_norm(norm, out_channels)
                self.norm2 = create_norm(norm, out_channels)

                self.act_fn1 = create_activation_fn(activation_fn, out_channels)
                self.act_fn2 = create_activation_fn(activation_fn, out_channels)

                if eql:
                    self.toRGB = nn.Sequential(
                        EqlConv2d(out_channels, image_channels, (1, 1), (1, 1), (0, 0)),
                        nn.Tanh()
                    )
                else:
                    self.toRGB = nn.Sequential(
                        nn.Conv2d(out_channels, image_channels, (1, 1), (1, 1), (0, 0)),
                        nn.Tanh()
                    )

            def forward(self, x):
                x = self.compute1(x)
                x = self.norm1(x)
                x = self.act_fn1(x)

                x = self.compute2(x)
                x = self.norm2(x)
                x = self.act_fn2(x)

                rgb = self.toRGB(x)
                return x, rgb

        class LastBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.compute1 = nn.Sequential(
                    create_upscale(upscale),
                    Conv(in_channels, out_channels),
                )
                # self.compute1 = create_upscale(upscale, in_channels, out_channels)
                self.compute2 = Conv(out_channels, out_channels)

                self.norm1 = create_norm(norm, out_channels)
                self.norm2 = create_norm(norm, out_channels)

                self.act_fn1 = create_activation_fn(activation_fn, out_channels)
                self.act_fn2 = create_activation_fn(activation_fn, out_channels)

                if eql:
                    self.toRGB = nn.Sequential(
                        EqlConv2d(out_channels, image_channels, (1, 1), (1, 1), (0, 0)),
                        nn.Tanh()
                    )
                else:
                    self.toRGB = nn.Sequential(
                        nn.Conv2d(out_channels, image_channels, (1, 1), (1, 1), (0, 0)),
                        nn.Tanh()
                    )

            def forward(self, x):
                x = self.compute1(x)
                x = self.norm1(x)
                x = self.act_fn1(x)

                x = self.compute2(x)
                x = self.norm2(x)
                x = self.act_fn2(x)

                rgb = self.toRGB(x)
                return x, rgb

        # END block declaration section

        self.channels = [
            latent_dim,
            *list(reversed([
                2 ** i * g_depth
                for i in range(2, int(math.log2(image_size)))
            ])),
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

    def forward(self, x):
        rgbs = []
        for b in self.blocks:
            x, rgb = b(x)
            rgbs.insert(0, rgb)

        return rgbs

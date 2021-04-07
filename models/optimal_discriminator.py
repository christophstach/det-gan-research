from torch import nn
from torch.nn.functional import interpolate
from torch.nn.utils import spectral_norm


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
                    spectral_norm(nn.Conv2d(in_channels, out_channels, (1, 1), (1, 1), (0, 0)))
                )

                self.compute1 = spectral_norm(nn.Conv2d(in_channels, in_channels, (3, 3), (1, 1), (1, 1)))
                self.act_fn1 = nn.LeakyReLU(0.2)

                self.compute2 = nn.Sequential(
                    Downscale(),
                    spectral_norm(nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1)))
                )
                self.act_fn2 = nn.LeakyReLU(0.2)

            def forward(self, x):
                identity = self.down(x)
                x = self.act_fn1(self.compute1(x))
                x = self.act_fn2(self.compute2(x))

                return x + identity

        class IntermediateBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.down = nn.Sequential(
                    Downscale(),
                    spectral_norm(nn.Conv2d(in_channels, out_channels, (1, 1), (1, 1), (0, 0)))
                )

                self.compute1 = spectral_norm(nn.Conv2d(in_channels, in_channels, (3, 3), (1, 1), (1, 1)))
                self.act_fn1 = nn.LeakyReLU(0.2)

                self.compute2 = nn.Sequential(
                    Downscale(),
                    spectral_norm(nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1)))
                )
                self.act_fn2 = nn.LeakyReLU(0.2)

            def forward(self, x):
                identity = self.down(x)
                x = self.act_fn1(self.compute1(x))
                x = self.act_fn2(self.compute2(x))

                return x + identity

        class LastBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.down = spectral_norm(nn.Conv2d(in_channels, out_channels, (4, 4), (1, 1), (0, 0)))

                self.compute1 = spectral_norm(nn.Conv2d(in_channels, in_channels, (3, 3), (1, 1), (1, 1)))
                self.act_fn1 = nn.LeakyReLU(0.2)

                self.compute2 = spectral_norm(nn.Conv2d(in_channels, out_channels, (4, 4), (1, 1), (0, 0)))
                self.act_fn2 = nn.LeakyReLU(0.2)

                self.scorer = spectral_norm(nn.Conv2d(out_channels, out_channels, (1, 1), (1, 1), (0, 0)))

            def forward(self, x):
                identity = self.down(x)
                x = self.act_fn1(self.compute1(x))
                x = self.act_fn2(self.compute2(x))
                x = self.scorer(x + identity)

                return x

        # END block declaration section

        channels = [
            image_channels,
            4 * d_depth,  # 8
            8 * d_depth,  # 16
            16 * d_depth,  # 32
            32 * d_depth,  # 64
            # 64 * d_depth,  # 128
            # 128 * d_depth,  # 256
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

        self.apply(weights_init)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)

        return x

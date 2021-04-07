from torch import nn
from torch.nn.functional import interpolate


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
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(in_channels, in_channels, (1, 1), (1, 1), (0, 0)),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(in_channels, in_channels, (1, 1), (1, 1), (0, 0)),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(in_channels, in_channels, (1, 1), (1, 1), (0, 0)),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(in_channels, in_channels, (1, 1), (1, 1), (0, 0)),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(in_channels, in_channels, (1, 1), (1, 1), (0, 0)),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(in_channels, in_channels, (1, 1), (1, 1), (0, 0)),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(in_channels, in_channels, (1, 1), (1, 1), (0, 0)),
                    nn.LeakyReLU(0.2)
                )

                self.compute1 = nn.ConvTranspose2d(in_channels, out_channels, (4, 4), (1, 1), (0, 0))
                self.act_fn1 = nn.LeakyReLU(0.2)
                self.norm1 = nn.BatchNorm2d(out_channels)

                self.compute2 = nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), (1, 1))
                self.act_fn2 = nn.LeakyReLU(0.2)
                self.norm2 = nn.BatchNorm2d(out_channels)

            def forward(self, x):
                # x = w = self.map(x)
                x = self.norm1(self.act_fn1(self.compute1(x)))
                x = self.norm2(self.act_fn2(self.compute2(x)))

                return x

        class IntermediateBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.compute1 = nn.Sequential(
                    Upscale(),
                    nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1))
                )
                self.act_fn1 = nn.LeakyReLU(0.2)
                self.norm1 = nn.BatchNorm2d(out_channels)

                self.compute2 = nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), (1, 1))
                self.act_fn2 = nn.LeakyReLU(0.2)
                self.norm2 = nn.BatchNorm2d(out_channels)

            def forward(self, x):
                x = self.norm1(self.act_fn1(self.compute1(x)))
                x = self.norm2(self.act_fn2(self.compute2(x)))

                return x

        class LastBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.compute1 = nn.Sequential(
                    Upscale(),
                    nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1))
                )
                self.act_fn1 = nn.LeakyReLU(0.2)
                self.norm1 = nn.BatchNorm2d(out_channels)

                self.compute2 = nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), (1, 1))
                self.act_fn2 = nn.LeakyReLU(0.2)
                self.norm2 = nn.BatchNorm2d(out_channels)

                self.toRGB = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, (1, 1), (1, 1), (0, 0)),
                    nn.Tanh()
                )

            def forward(self, x):
                x = self.norm1(self.act_fn1(self.compute1(x)))
                x = self.norm2(self.act_fn2(self.compute2(x)))
                x = self.toRGB(x)

                print(x.shape)

                return x

        # END block declaration section

        channels = [
            latent_dimension,
            # 128 * g_depth,  # 256
            # 64 * g_depth,  # 128
            32 * g_depth,  # 64
            16 * g_depth,  # 32
            8 * g_depth,  # 16
            4 * g_depth,  # 8
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

        self.apply(weights_init)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)

        return x

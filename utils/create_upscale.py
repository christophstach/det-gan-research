from torch import nn
from torch.nn.functional import interpolate


def create_upscale(upscale: str, in_channels: int = None, out_channels: int = None):
    class Upscale(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return interpolate(
                input=x,
                scale_factor=(2, 2),
                mode='bilinear',
                align_corners=False,
            )

    upscale_dict = {
        "interpolate": lambda: Upscale(),
        "shuffle": lambda: nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, (1, 1), (1, 1), (0, 0)),
            nn.PixelUnshuffle(2),
        ),
        "conv": lambda: nn.ConvTranspose2d(in_channels, out_channels, (4, 4), (2, 2), (1, 1))
    }

    return upscale_dict[upscale]()

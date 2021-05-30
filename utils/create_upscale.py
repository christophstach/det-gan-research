from torch import nn
from torch.nn.functional import interpolate

from utils.icnr import ConvShuffle


def create_upscale(upscale: str, in_channels: int = None, out_channels: int = None):
    class NearestUpscale(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return interpolate(
                input=x,
                scale_factor=(2, 2),
                mode='nearest'
            )

    class BilinearUpscale(nn.Module):
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
        'nearest': lambda: NearestUpscale(),
        'bilinear': lambda: BilinearUpscale(),
        'shuffle': lambda: nn.PixelShuffle(2),  # channels size divided by 4
        'conv_shuffle': lambda: ConvShuffle(in_channels, out_channels, upscale=2),
        'deconv': lambda: nn.ConvTranspose2d(
            in_channels,
            out_channels,
            (4, 4),
            (2, 2),
            (1, 1),
            padding_mode='zeros'
        )

    }

    return upscale_dict[upscale]()

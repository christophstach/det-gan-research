from torch import nn
from torch.nn.functional import interpolate
from torch.nn.utils import spectral_norm as sn


def create_downscale(downscale: str, in_channels: int = None, out_channels: int = None):
    class NearestDownscale(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return interpolate(
                input=x,
                scale_factor=(0.5, 0.5),
                mode='nearest'
            )

    class BilinearDownscale(nn.Module):
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

    downscale_dict = {
        'maxpool': lambda: nn.MaxPool2d(2, 2),
        'avgpool': lambda: nn.AvgPool2d(2, 2),
        'nearest': lambda: NearestDownscale(),
        'bilinear': lambda: BilinearDownscale(),
        'unshuffle': lambda: nn.Sequential(
            sn(nn.Conv2d(in_channels, out_channels // 4, (1, 1), (1, 1), (0, 0))),
            nn.PixelUnshuffle(2),
        ),
        'conv': lambda: sn(
            nn.Conv2d(in_channels, out_channels, (4, 4), (2, 2), (1, 1), padding_mode='replicate')
        )
    }

    return downscale_dict[downscale]()

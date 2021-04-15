import layers as l
from torch import nn


def create_norm(norm: str, num_features: int):
    norm_dict = {
        "passthrough": lambda: l.Passthrough(),
        "pixel": lambda: l.PixelNorm(),
        "batch": lambda: nn.BatchNorm2d(num_features),
        "switchable": lambda: l.SwitchNorm2d(num_features),
        "instance": lambda: nn.InstanceNorm2d(num_features, affine=True),
        "sparse_switchable": lambda: l.SparseSwitchNorm2d(num_features)
    }

    return norm_dict[norm]()

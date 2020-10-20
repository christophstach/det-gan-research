import torch
from torch.utils.data import TensorDataset


def noise(length, shape, uniform=False):
    if uniform:
        return TensorDataset(
            torch.rand([length, *shape])
        )
    else:
        return TensorDataset(
            torch.randn([length, *shape])
        )

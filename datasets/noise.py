import torch
from torch.utils.data import TensorDataset


def noise(length, shape):
    return TensorDataset(
        torch.rand([length, *shape])
    )

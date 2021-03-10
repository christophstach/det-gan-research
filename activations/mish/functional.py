"""
https://github.com/digantamisra98/Mish/tree/master/Mish/Torch/functional.py

Script provides functional interface for Mish activation function.
"""

# import pytorch
import torch
import torch.nn.functional as F


@torch.jit.script
def mish(x):
    """
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    See additional documentation for mish class.
    """
    return x * torch.tanh(F.softplus(x))

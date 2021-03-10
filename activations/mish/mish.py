"""
https://github.com/digantamisra98/Mish/tree/master/Mish/Torch/mish.py

Applies the mish function element-wise:
mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
"""

# import pytorch
import torch
from torch import nn

# import activation functions
import activations.mish.functional as F


class Mish(nn.Module):
    """
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Examples:
        >>> m = Mish()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    def __init__(self):
        """
        Init method.
        """
        super().__init__()

    def forward(self, x):
        """
        Forward pass of the function.
        """
        return F.mish(x)

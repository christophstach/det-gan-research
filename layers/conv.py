import collections.abc
from itertools import repeat
from typing import Union, TypeVar, Tuple

import torch.nn.functional as F
from numpy import sqrt, prod
from torch import nn, empty, FloatTensor
from torch.nn import init


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


def _reverse_repeat_tuple(t, n):
    r"""Reverse the order of `t` and repeat each element for `n` times.

    This can be used to translate padding arg used by Conv and Pooling modules
    to the ones used by `F.pad`.
    """
    return tuple(x for x in reversed(t) for _ in range(n))


T = TypeVar('T')
_scalar_or_tuple_any_t = Union[T, Tuple[T, ...]]
_scalar_or_tuple_1_t = Union[T, Tuple[T]]
_scalar_or_tuple_2_t = Union[T, Tuple[T, T]]
_scalar_or_tuple_3_t = Union[T, Tuple[T, T, T]]
_scalar_or_tuple_4_t = Union[T, Tuple[T, T, T, T]]
_scalar_or_tuple_5_t = Union[T, Tuple[T, T, T, T, T]]
_scalar_or_tuple_6_t = Union[T, Tuple[T, T, T, T, T, T]]

# For arguments which represent size parameters (eg, kernel size, padding)
_size_any_t = _scalar_or_tuple_any_t[int]
_size_1_t = _scalar_or_tuple_1_t[int]
_size_2_t = _scalar_or_tuple_2_t[int]
_size_3_t = _scalar_or_tuple_3_t[int]
_size_4_t = _scalar_or_tuple_4_t[int]
_size_5_t = _scalar_or_tuple_5_t[int]
_size_6_t = _scalar_or_tuple_6_t[int]


class EqlConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 stride: _size_2_t = 1,
                 padding: _size_2_t = 0,
                 dilation: _size_2_t = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros'):

        super().__init__()

        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.padding_mode = padding_mode
        self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)

        # define the weight and bias if to be used
        self.weight = nn.Parameter(
            init.normal_(
                empty(out_channels, in_channels, *self.kernel_size)
            )
        )

        if bias:
            self.bias = nn.Parameter(FloatTensor(out_channels).fill_(0))

        fan_in = prod(self.kernel_size) * in_channels  # value of fan_in
        self.scale = sqrt(2 / sqrt(fan_in))

    def forward(self, x):
        if self.padding_mode != 'zeros':
            return F.conv2d(
                F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                self.weight * self.scale,
                self.bias,
                self.stride,
                _pair(0),
                self.dilation,
                self.groups
            )
        return F.conv2d(
            x,
            self.weight * self.scale,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )

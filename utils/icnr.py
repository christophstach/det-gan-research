import torch

from torch import nn


def icnr(tensor, factor=2, initializer=nn.init.kaiming_normal_):
    """Fills the input Tensor or Variable with values according to the method
    described in "Checkerboard artifact free sub-pixel convolution"
    - Andrew Aitken et al. (2017), this inizialization should be used in the
    last convolutional layer before a PixelShuffle operation
    Args:
        tensor: an n-dimensional torch.Tensor or autograd.Variable
        factor: factor to increase spatial resolution by
        initializer: initializer to be used for sub_kernel inizialization
    Examples:
        >>> upscale = 8
        >>> num_classes = 10
        >>> previous_layer_features = Variable(torch.Tensor(8, 64, 32, 32))
        >>> conv_shuffle = Conv2d(64, num_classes * (upscale ** 2), 3, padding=1, bias=0)
        >>> ps = PixelShuffle(upscale)
        >>> kernel = icnr(conv_shuffle.weight, factor=upscale)
        >>> conv_shuffle.weight.data.copy_(kernel)
        >>> output = ps(conv_shuffle(previous_layer_features))
        >>> print(output.shape)
        torch.Size([8, 10, 256, 256])
    .. _Checkerboard artifact free sub-pixel convolution:
        https://arxiv.org/abs/1707.02937
    """
    new_shape = [int(tensor.shape[0] / (factor ** 2))] + list(tensor.shape[1:])
    sub_kernel = torch.zeros(new_shape)
    sub_kernel = initializer(sub_kernel)
    sub_kernel = sub_kernel.transpose(0, 1)

    sub_kernel = sub_kernel.contiguous().view(sub_kernel.shape[0], sub_kernel.shape[1], -1)

    kernel = sub_kernel.repeat(1, 1, factor ** 2)

    transposed_shape = [tensor.shape[1]] + [tensor.shape[0]] + list(tensor.shape[2:])
    kernel = kernel.contiguous().view(transposed_shape)

    kernel = kernel.transpose(0, 1)

    return kernel


class ConvShuffle(nn.Module):
    def __init__(self, in_channels, out_channels, upscale=2):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels * (upscale ** 2),
            (1, 1),
            (1, 1),
            (0, 0),
            bias=False
        )
        kernel = icnr(self.conv.weight, factor=upscale)
        self.conv.weight.data.copy_(kernel)

        self.shuffle = nn.PixelShuffle(upscale)

    def forward(self, x):
        x = self.conv(x)
        x = self.shuffle(x)

        return x

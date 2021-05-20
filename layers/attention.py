import math
import torch
from torch import nn, bmm, softmax, transpose, Tensor, zeros
from torch.nn import functional as F
from torch.nn.utils.spectral_norm import spectral_norm as sn


class SelfAttention2d(nn.Module):
    def __init__(self, in_channels, out_channels, channel_divisor: int = 8, spectral_norm=False):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.f_g_channels = in_channels // channel_divisor

        self.gamma = nn.Parameter(zeros(1))

        self.f = nn.Conv2d(self.in_channels, self.f_g_channels, (1, 1), (1, 1), (0, 0))
        self.g = nn.Conv2d(self.in_channels, self.f_g_channels, (1, 1), (1, 1), (0, 0))
        self.h = nn.Conv2d(self.in_channels, self.in_channels, (1, 1), (1, 1), (0, 0))
        self.v = nn.Conv2d(self.in_channels, self.out_channels, (1, 1), (1, 1), (0, 0))

        if spectral_norm:
            self.f = sn(self.f)
            self.g = sn(self.g)
            self.h = sn(self.h)
            self.v = sn(self.v)

    def forward(self, x: Tensor):
        batch_size, channels, width, height = x.shape

        f = self.f(x).view(x.shape[0], self.f_g_channels, width * height)
        g = self.g(x).view(x.shape[0], self.f_g_channels, width * height)
        h = self.h(x).view(x.shape[0], self.in_channels, width * height)

        attention = softmax(bmm(transpose(f, 2, 1), g), dim=1)

        o = bmm(h, attention)
        o = o.view(batch_size, channels, width, height)

        v = self.v(self.gamma * o + x)

        return v


class EfficientChannelAttention(nn.Module):
    # https://github.com/BangguWu/ECANet/blob/master/models/eca_resnet.py
    # Added value and gamma (not existent in original application)

    def __init__(self, in_channels, out_channels, gamma=2, b=1, spectral_norm=False):
        super().__init__()

        t = int(abs((math.log2(in_channels) + b) / gamma))
        k = t if t % 2 else t + 1

        self.gamma = nn.Parameter(zeros(1))

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        if spectral_norm:
            self.conv = sn(
                nn.Conv1d(
                    in_channels=1,
                    out_channels=1,
                    kernel_size=k,
                    padding=int(k / 2),
                    padding_mode='zeros',
                    bias=False
                )
            )

            self.value = sn(
                nn.Conv2d(in_channels, out_channels, (1, 1), (1, 1), (0, 0))
            )
        else:
            self.conv = nn.Conv1d(
                in_channels=1,
                out_channels=1,
                kernel_size=k,
                padding=int(k / 2),
                padding_mode='zeros',
                bias=False
            )

            self.value = nn.Conv2d(in_channels, out_channels, (1, 1), (1, 1), (0, 0))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        v = self.value(x + (x * y.expand_as(x) * self.gamma))

        return v


class EfficientAttention(nn.Module):
    # https://github.com/cmsflash/efficient-attention/blob/master/efficient_attention.py
    # just added the option for spectral norm
    def __init__(self, in_channels, key_channels, head_count, value_channels, spectral_norm=False):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        if spectral_norm:
            self.keys = sn(nn.Conv2d(in_channels, key_channels, (1, 1)))
            self.queries = sn(nn.Conv2d(in_channels, key_channels, (1, 1)))
            self.values = sn(nn.Conv2d(in_channels, value_channels, (1, 1)))
            self.reprojection = sn(nn.Conv2d(value_channels, in_channels, (1, 1)))
        else:
            self.keys = nn.Conv2d(in_channels, key_channels, (1, 1))
            self.queries = nn.Conv2d(in_channels, key_channels, (1, 1))
            self.values = nn.Conv2d(in_channels, value_channels, (1, 1))
            self.reprojection = nn.Conv2d(value_channels, in_channels, (1, 1))

    def forward(self, input_):
        n, _, h, w = input_.size()
        keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        values = self.values(input_).reshape((n, self.value_channels, h * w))
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=2)
            query = F.softmax(queries[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=1)
            value = values[:, i * head_value_channels: (i + 1) * head_value_channels, :]
            context = key @ value.transpose(1, 2)
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, h, w)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        reprojected_value = self.reprojection(aggregated_values)
        attention = reprojected_value + input_

        return attention

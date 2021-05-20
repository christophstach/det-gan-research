import torch
import torch.nn.functional as F
from torch import nn


class ModConv(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim, bias=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.style_dim = style_dim
        self.kernel_size = 3

        self.weight = nn.Parameter(
            nn.init.normal_(
                torch.empty(1, self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
            )
        )

        if bias:
            self.bias = nn.Parameter(
                nn.init.uniform_(
                    torch.empty(self.out_channels)
                )
            )
        else:
            self.register_parameter('bias', None)

        self.modulation = nn.Linear(style_dim, in_channels)
        self.epsilon = 1e-8
        self.demodulation = True

    def forward(self, x, w):
        batch_size, channels, height, width = x.shape
        style = self.modulation(w)
        styled_weight = self.weight * style.view(batch_size, 1, self.in_channels, 1, 1)
        bias = self.bias.repeat(batch_size)

        if self.demodulation:
            sigma_weight = torch.rsqrt(styled_weight.square().sum([2, 3, 4]).add(self.epsilon))
            styled_weight = styled_weight * sigma_weight.view(batch_size, self.out_channels, 1, 1, 1)

        weight = styled_weight.view(
            batch_size * self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size
        )

        x = x.view(1, batch_size * self.in_channels, height, width)

        out = F.conv2d(
            input=F.pad(x, (1, 1, 1, 1), 'reflect'),
            weight=weight,
            # bias=bias,
            groups=batch_size
        )

        return out.view(batch_size, self.out_channels, height, width)

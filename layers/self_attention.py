from torch import nn, bmm, softmax, transpose, Tensor, zeros

from torch.nn.utils.spectral_norm import spectral_norm as sn


class SelfAttention2d(nn.Module):
    def __init__(self, in_channels, out_channels, channel_divisor: int = 8, spectral_norm=False):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.f_g_channels = in_channels // channel_divisor

        # self.gamma = nn.Parameter(zeros(1))

        self.f = nn.Conv2d(self.in_channels, self.f_g_channels, (1, 1), (1, 1), (0, 0))
        self.g = nn.Conv2d(self.in_channels, self.f_g_channels, (1, 1), (1, 1), (0, 0))
        self.h = nn.Conv2d(self.in_channels, self.in_channels, (1, 1), (1, 1), (0, 0))
        self.v = nn.Conv2d(self.in_channels, self.out_channels, (1, 1), (1, 1), (0, 0))

        if spectral_norm:
            self.f = sn(self.f)
            self.g = sn(self.g)
            self.h = sn(self.h)

    def forward(self, x: Tensor):
        batch_size, channels, width, height = x.shape

        f = self.f(x).view(x.shape[0], self.f_g_channels, width * height)
        g = self.g(x).view(x.shape[0], self.f_g_channels, width * height)
        h = self.h(x).view(x.shape[0], self.in_channels, width * height)

        attention = softmax(bmm(transpose(f, 2, 1), g), dim=1)

        o = bmm(h, attention)
        o = o.view(batch_size, channels, width, height)

        v = self.v(o + x)

        return v

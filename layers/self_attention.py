from torch import nn, bmm, softmax, transpose, Tensor, zeros

from torch.nn.utils.spectral_norm import spectral_norm as sn


class SelfAttention2d(nn.Module):
    def __init__(self, channels, channel_divisor: int = 8, spectral_norm=False):
        super().__init__()

        self.channels = channels
        self.f_g_channels = channels // channel_divisor

        self.gamma = nn.Parameter(zeros(1))

        self.f = nn.Conv2d(self.channels, self.f_g_channels, (1, 1), (1, 1), (0, 0))
        self.g = nn.Conv2d(self.channels, self.f_g_channels, (1, 1), (1, 1), (0, 0))
        self.h = nn.Conv2d(self.channels, self.channels, (1, 1), (1, 1), (0, 0))

        if spectral_norm:
            self.f = sn(self.f)
            self.g = sn(self.g)
            self.h = sn(self.h)

    def forward(self, x: Tensor):
        batch_size, channels, width, height = x.shape

        f = self.f(x).view(x.shape[0], self.f_g_channels, width * height)
        g = self.g(x).view(x.shape[0], self.f_g_channels, width * height)
        h = self.h(x).view(x.shape[0], self.channels, width * height)

        attention = softmax(bmm(transpose(f, 2, 1), g), dim=1)

        o = bmm(h, attention)
        o = o.view(batch_size, channels, width, height)

        return self.gamma * o + x, attention

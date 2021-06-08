import math

from torch import nn


class OctaveConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 alpha_in=0.5,
                 alpha_out=0.5,
                 stride=(1, 1),
                 padding=(0, 0),
                 dilation=(1, 1),
                 groups=1,
                 bias=False,
                 padding_mode: str = 'zeros'):
        super(OctaveConv, self).__init__()

        assert stride == 1 or stride == 2 or stride == (1, 1) or stride == (2, 2), "Stride should be 1 or 2."
        assert 0 <= alpha_in <= 1 and 0 <= alpha_out <= 1, "Alphas should be in the interval from 0 to 1."

        self.down = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        self.stride = stride
        self.is_dw = groups == in_channels

        self.alpha_in = alpha_in
        self.alpha_out = alpha_out

        self.conv_l2l = None if alpha_in == 0 or alpha_out == 0 else nn.Conv2d(
            int(alpha_in * in_channels),
            int(alpha_out * out_channels),
            kernel_size,
            (1, 1),
            padding,
            dilation,
            math.ceil(alpha_in * groups),
            bias,
            padding_mode
        )

        self.conv_l2h = None if alpha_in == 0 or alpha_out == 1 or self.is_dw else nn.Conv2d(
            int(alpha_in * in_channels),
            out_channels - int(alpha_out * out_channels),
            kernel_size,
            (1, 1),
            padding,
            dilation,
            groups,
            bias,
            padding_mode
        )

        self.conv_h2l = None if alpha_in == 1 or alpha_out == 0 or self.is_dw else nn.Conv2d(
            in_channels - int(alpha_in * in_channels),
            int(alpha_out * out_channels),
            kernel_size,
            (1, 1),
            padding,
            dilation,
            groups,
            bias,
            padding_mode
        )

        self.conv_h2h = None if alpha_in == 1 or alpha_out == 1 else nn.Conv2d(
            in_channels - int(alpha_in * in_channels),
            out_channels - int(alpha_out * out_channels),
            kernel_size,
            (1, 1),
            padding,
            dilation,
            math.ceil(groups - alpha_in * groups),
            bias,
            padding_mode
        )

    def forward(self, x):
        x_h, x_l = x if type(x) is tuple else (x, None)

        x_h = self.down(x_h) if self.stride == 2 else x_h
        x_h2h = self.conv_h2h(x_h)
        x_h2l = self.conv_h2l(self.down(x_h)) if self.alpha_out > 0 and not self.is_dw else None

        if x_l is not None:
            x_l2l = self.down(x_l) if self.stride == 2 else x_l
            x_l2l = self.conv_l2l(x_l2l) if self.alpha_out > 0 else None

            if self.is_dw:
                return x_h2h, x_l2l
            else:
                x_l2h = self.conv_l2h(x_l)
                x_l2h = self.up(x_l2h) if self.stride == 1 else x_l2h
                x_h = x_l2h + x_h2h
                x_l = x_h2l + x_l2l if x_h2l is not None and x_l2l is not None else None
                return x_h, x_l
        else:
            return x_h2h, x_h2l

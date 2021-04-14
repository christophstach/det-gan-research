from torch import nn

from utils import create_activation_fn


class StyleDisentangler(nn.Module):
    def __init__(self, latent_dim, m_depth=8):
        super().__init__()

        activation_fn = 'lrelu'

        self.m_depth = m_depth
        self.main = nn.ModuleList()

        for i in range(m_depth):
            self.main.append(
                nn.Sequential(
                    nn.Conv2d(latent_dim, latent_dim, (1, 1), (1, 1), (0, 0)),
                    create_activation_fn(activation_fn, latent_dim)
                )
            )

    def forward(self, z):
        w = z

        for b in self.main:
            w = b(w)

        return w

from torch import nn


class UnaryDiscriminator(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()

        self.model = model

    def forward(self, x):
        x = self.model(x)
        return x.view(-1, 2)

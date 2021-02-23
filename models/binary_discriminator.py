from torch import nn


class BinaryDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(2, 16),
            nn.SELU(inplace=True),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        x = self.main(x)
        return x.view(-1, 1).squeeze(1)
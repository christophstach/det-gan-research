import torch
import torch.nn.functional as F

import losses.base


class Hinge(losses.base.Loss):
    def __init__(self) -> None:
        super().__init__()

    def discriminator_loss(self, real_validity: torch.Tensor, fake_validity: torch.Tensor) -> torch.Tensor:
        real_loss = torch.relu(1.0 - real_validity)
        fake_loss = torch.relu(1.0 + fake_validity)

        loss = real_loss.mean() + fake_loss.mean()

        return loss.unsqueeze(0)

    def generator_loss(self, real_validity: torch.Tensor, fake_validity: torch.Tensor) -> torch.Tensor:
        fake_loss = -fake_validity

        loss = fake_loss.mean()

        return loss.unsqueeze(0)



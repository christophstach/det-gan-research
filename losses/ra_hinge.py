import torch

import losses.base


class RaHinge(losses.base.Loss):
    def __init__(self) -> None:
        super().__init__()

    def discriminator_loss(self, real_validity: torch.Tensor, fake_validity: torch.Tensor) -> torch.Tensor:
        relativistic_real_validity = real_validity - fake_validity.mean()
        relativistic_fake_validity = fake_validity - real_validity.mean()

        real_loss = torch.relu(1.0 - relativistic_real_validity)
        fake_loss = torch.relu(1.0 + relativistic_fake_validity)

        loss = (real_loss.mean() + fake_loss.mean()) / 2

        return loss.unsqueeze(0)

    def generator_loss(self, real_validity: torch.Tensor, fake_validity: torch.Tensor) -> torch.Tensor:
        relativistic_real_validity = real_validity - fake_validity.mean()
        relativistic_fake_validity = fake_validity - real_validity.mean()

        real_loss = torch.relu(1.0 - relativistic_fake_validity)
        fake_loss = torch.relu(1.0 + relativistic_real_validity)

        loss = (fake_loss.mean() + real_loss.mean()) / 2

        return loss.unsqueeze(0)



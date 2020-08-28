import torch

import losses.base


class RaLSGAN(losses.base.Loss):
    def __init__(self) -> None:
        super().__init__()

    def discriminator_loss(self, real_validity: torch.Tensor, fake_validity: torch.Tensor) -> torch.Tensor:
        relativistic_real_validity = real_validity - fake_validity.mean()
        relativistic_fake_validity = fake_validity - real_validity.mean()

        real_loss = (relativistic_real_validity - 1.0) ** 2
        fake_loss = (relativistic_fake_validity + 1.0) ** 2

        loss = (fake_loss.mean() + real_loss.mean()) / 2

        return loss.unsqueeze(0)

    def generator_loss(self, real_validity: torch.Tensor, fake_validity: torch.Tensor) -> torch.Tensor:
        relativistic_real_validity = real_validity - fake_validity.mean()
        relativistic_fake_validity = fake_validity - real_validity.mean()

        real_loss = (relativistic_real_validity + 1.0) ** 2
        fake_loss = (relativistic_fake_validity - 1.0) ** 2

        loss = (fake_loss.mean() + real_loss.mean()) / 2

        return loss.unsqueeze(0)







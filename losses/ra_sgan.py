import torch
import torch.nn.functional as F

import losses.base


class RaSGAN(losses.base.Loss):
    def __init__(self) -> None:
        super().__init__()

    def discriminator_loss(self, real_validity: torch.Tensor, fake_validity: torch.Tensor) -> torch.Tensor:
        relativistic_real_validity = real_validity - fake_validity.mean()
        relativistic_fake_validity = fake_validity - real_validity.mean()

        real_label = torch.ones_like(real_validity)
        fake_label = torch.zeros_like(fake_validity)

        relativistic_real_probability = F.binary_cross_entropy_with_logits(relativistic_real_validity, real_label)
        relativistic_fake_probability = F.binary_cross_entropy_with_logits(relativistic_fake_validity, fake_label)

        loss = (relativistic_real_probability + relativistic_fake_probability) / 2

        return loss.unsqueeze(0)

    def generator_loss(self, real_validity: torch.Tensor, fake_validity: torch.Tensor) -> torch.Tensor:
        relativistic_real_validity = real_validity - fake_validity.mean()
        relativistic_fake_validity = fake_validity - real_validity.mean()

        real_label = torch.ones_like(real_validity)
        fake_label = torch.zeros_like(fake_validity)

        relativistic_real_probability = F.binary_cross_entropy_with_logits(relativistic_real_validity, fake_label)
        relativistic_fake_probability = F.binary_cross_entropy_with_logits(relativistic_fake_validity, real_label)

        loss = (relativistic_real_probability + relativistic_fake_probability) / 2

        return loss.unsqueeze(0)

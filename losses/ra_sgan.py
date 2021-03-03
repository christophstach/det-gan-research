import torch
import torch.nn.functional as F

import losses.base


class RaSGAN(losses.base.Loss):
    def __init__(self) -> None:
        super().__init__()

    def discriminator_loss(self, real_scores: torch.Tensor, fake_scores: torch.Tensor) -> torch.Tensor:
        relativistic_real_scores = real_scores - fake_scores.mean()
        relativistic_fake_scores = fake_scores - real_scores.mean()

        real_label = torch.ones_like(real_scores)
        fake_label = torch.zeros_like(fake_scores)

        relativistic_real_probability = F.binary_cross_entropy_with_logits(relativistic_real_scores, real_label)
        relativistic_fake_probability = F.binary_cross_entropy_with_logits(relativistic_fake_scores, fake_label)

        loss = (relativistic_real_probability + relativistic_fake_probability) / 2

        return loss.unsqueeze(0)

    def generator_loss(self, real_scores: torch.Tensor, fake_scores: torch.Tensor) -> torch.Tensor:
        relativistic_real_scores = real_scores - fake_scores.mean()
        relativistic_fake_scores = fake_scores - real_scores.mean()

        real_label = torch.ones_like(real_scores)
        fake_label = torch.zeros_like(fake_scores)

        relativistic_real_probability = F.binary_cross_entropy_with_logits(relativistic_real_scores, fake_label)
        relativistic_fake_probability = F.binary_cross_entropy_with_logits(relativistic_fake_scores, real_label)

        loss = (relativistic_real_probability + relativistic_fake_probability) / 2

        return loss

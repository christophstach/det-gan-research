import torch

import losses.base


class RaLSGAN(losses.base.Loss):
    def __init__(self) -> None:
        super().__init__()

    def discriminator_loss(self, real_scores: torch.Tensor, fake_scores: torch.Tensor) -> torch.Tensor:
        relativistic_real_scores = real_scores - fake_scores.mean()
        relativistic_fake_scores = fake_scores - real_scores.mean()

        real_loss = (relativistic_real_scores - 1.0) ** 2
        fake_loss = (relativistic_fake_scores + 1.0) ** 2

        loss = (fake_loss.mean() + real_loss.mean()) / 2

        return loss.unsqueeze(0)

    def generator_loss(self, real_scores: torch.Tensor, fake_scores: torch.Tensor) -> torch.Tensor:
        relativistic_real_scores = real_scores - fake_scores.mean()
        relativistic_fake_scores = fake_scores - real_scores.mean()

        real_loss = (relativistic_real_scores + 1.0) ** 2
        fake_loss = (relativistic_fake_scores - 1.0) ** 2

        loss = (fake_loss.mean() + real_loss.mean()) / 2

        return loss







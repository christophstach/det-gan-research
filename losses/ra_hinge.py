import torch

import losses.base


class RaHinge(losses.base.Loss):
    def __init__(self) -> None:
        super().__init__()

    def discriminator_loss(self, real_scores: torch.Tensor, fake_scores: torch.Tensor) -> torch.Tensor:
        relativistic_real_validity = real_scores - fake_scores.mean()
        relativistic_fake_validity = fake_scores - real_scores.mean()

        real_loss = torch.relu(1.0 - relativistic_real_validity)
        fake_loss = torch.relu(1.0 + relativistic_fake_validity)

        loss = (real_loss.mean() + fake_loss.mean()) / 2

        return loss

    def generator_loss(self, real_scores: torch.Tensor, fake_scores: torch.Tensor) -> torch.Tensor:
        relativistic_real_validity = real_scores - fake_scores.mean()
        relativistic_fake_validity = fake_scores - real_scores.mean()

        real_loss = torch.relu(1.0 - relativistic_fake_validity)
        fake_loss = torch.relu(1.0 + relativistic_real_validity)

        loss = (fake_loss.mean() + real_loss.mean()) / 2

        return loss



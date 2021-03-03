import torch

import losses.base


class Hinge(losses.base.Loss):
    def __init__(self) -> None:
        super().__init__()

    def discriminator_loss(self, real_scores: torch.Tensor, fake_scores: torch.Tensor) -> torch.Tensor:
        real_loss = torch.relu(1.0 - real_scores)
        fake_loss = torch.relu(1.0 + fake_scores)

        loss = real_loss.mean() + fake_loss.mean()

        return loss

    def generator_loss(self, real_scores: torch.Tensor, fake_scores: torch.Tensor) -> torch.Tensor:
        fake_loss = -fake_scores

        loss = fake_loss.mean()

        return loss

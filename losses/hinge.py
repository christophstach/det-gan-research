import torch
from torch import Tensor

import losses.base


class Hinge(losses.base.Loss):
    def __init__(self) -> None:
        super().__init__()

    def discriminator_loss(self, real_scores: Tensor, fake_scores: Tensor) -> Tensor:
        real_loss = torch.relu(1.0 - real_scores)
        fake_loss = torch.relu(1.0 + fake_scores)

        loss = real_loss.mean() + fake_loss.mean()

        return loss

    def generator_loss(self, real_scores: Tensor, fake_scores: Tensor) -> Tensor:
        fake_loss = -fake_scores

        loss = fake_loss.mean()

        return loss

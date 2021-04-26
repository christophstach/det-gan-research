import torch
from numpy import random, histogram, array
from torch import Tensor, from_numpy

import losses.base.loss


class Realness(losses.base.Loss):
    def __init__(self, num_outcomes=32) -> None:
        super().__init__()

        self.num_outcomes = num_outcomes

        gauss = random.normal(0.0, 0.1, 10000)
        count, _ = histogram(gauss, self.num_outcomes)
        self.real_anchor = from_numpy(array(count / sum(count))).float()

        uniform = random.uniform(-1.0, 1.0, 10000)
        count, _ = histogram(uniform, self.num_outcomes)
        self.fake_anchor = from_numpy(array(count / sum(count))).float()

    def kl_div(self, p, q):
        return torch.mean(torch.sum(p * (p / q).log(), dim=1))

    def discriminator_loss(self, real_scores: Tensor, fake_scores: Tensor) -> Tensor:
        self.real_anchor = self.real_anchor.to(real_scores.device)
        self.fake_anchor = self.real_anchor.to(real_scores.device)

        real_loss = self.kl_div(real_scores, self.fake_anchor)
        fake_loss = self.kl_div(self.real_anchor, fake_scores)
        loss = real_loss + fake_loss

        return loss

    def generator_loss(self, real_scores: Tensor, fake_scores: Tensor) -> Tensor:
        self.real_anchor = self.real_anchor.to(real_scores.device)
        self.fake_anchor = self.real_anchor.to(real_scores.device)

        # No relativism
        # loss = self.kl_div(self.fake_anchor, fake_scores)

        # EQ19_V1
        # loss = self.kl_div(self.fake_anchor, fake_scores) + self.kl_div(real_scores, fake_scores)

        # EQ19_V2 (default)
        loss = -self.kl_div(self.real_anchor, fake_scores) + self.kl_div(real_scores, fake_scores)

        # EQ20
        # loss = self.kl_div(self.fake_anchor, fake_scores) - self.kl_div(self.real_anchor, fake_scores)

        return loss

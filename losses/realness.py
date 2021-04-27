import torch
from numpy import histogram, random
from scipy.stats import skewnorm
from torch import Tensor, from_numpy

import losses.base.loss


def js_div(p, q, reduce=True):
    m = 0.5 * (p + q)
    jsd = 0.5 * (kl_div(p, m, reduce=False) + kl_div(q, m, reduce=False))

    return torch.mean(jsd) if reduce else jsd


def kl_div(p, q, epsilon=1e-12, reduce=True):
    kld = torch.sum(
        p * (p / (q + epsilon)).log(),
        dim=1
    )

    return torch.mean(kld) if reduce else kld


class Realness(losses.base.Loss):
    def __init__(self, score_dim) -> None:
        super().__init__()

        self.score_dim = score_dim
        self.gauss_uniform = True
        self.measure = 'kl'

        if self.measure == 'js':
            self.div = js_div
        elif self.measure == 'kl':
            self.div = kl_div
        else:
            raise NotImplementedError()

        if self.gauss_uniform:
            gauss = random.normal(0.0, 0.1, size=10000)
            count, _ = histogram(gauss, self.score_dim)
            self.anchor0 = from_numpy(count / sum(count)).float()

            uniform = random.uniform(-1.0, 1.0, size=10000)
            count, _ = histogram(uniform, self.score_dim)
            self.anchor1 = from_numpy(count / sum(count)).float()
        else:
            skew_left = skewnorm.rvs(-5, size=10000)
            count, _ = histogram(skew_left, self.score_dim)
            self.anchor0 = from_numpy(count / sum(count)).float()

            skew_right = skewnorm.rvs(5, size=10000)
            count, _ = histogram(skew_right, self.score_dim)
            self.anchor1 = from_numpy(count / sum(count)).float()

    def discriminator_loss(self, real_scores: Tensor, fake_scores: Tensor) -> Tensor:
        self.anchor0 = self.anchor0.to(real_scores.device)
        self.anchor1 = self.anchor1.to(real_scores.device)

        loss = self.div(self.anchor1, real_scores) + self.div(self.anchor0, fake_scores)
        # loss -= self.div(self.anchor1, fake_scores) + self.div(self.anchor0, real_scores)

        return loss

    def generator_loss(self, real_scores: Tensor, fake_scores: Tensor) -> Tensor:
        self.anchor0 = self.anchor0.to(real_scores.device)
        self.anchor1 = self.anchor1.to(real_scores.device)

        # No relativism
        # loss = self.div(self.anchor0, fake_scores)

        # EQ19 (default)
        loss = self.div(real_scores, fake_scores) - self.div(self.anchor0, fake_scores)

        # EQ20
        # loss = self.div(self.anchor1, fake_scores) - self.div(self.anchor0, fake_scores)

        return loss

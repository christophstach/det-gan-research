import torch
from numpy import histogram, random
from scipy.stats import skewnorm
from torch import Tensor, from_numpy
from torch.nn.functional import softmax

import losses.base.loss
from utils.stat import torch_wasserstein_loss


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
            self.distance = js_div
        elif self.measure == 'kl':
            self.distance = kl_div
        elif self.measure == 'emd':
            self.distance = torch_wasserstein_loss
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
            skew_left = skewnorm.rvs(-1.0, size=10000)
            count, _ = histogram(skew_left, self.score_dim)
            self.anchor0 = from_numpy(count / sum(count)).float()

            skew_right = skewnorm.rvs(1.0, size=10000)
            count, _ = histogram(skew_right, self.score_dim)
            self.anchor1 = from_numpy(count / sum(count)).float()

    def discriminator_loss(self, real_scores: Tensor, fake_scores: Tensor) -> Tensor:
        self.anchor0 = self.anchor0.to(real_scores)
        self.anchor1 = self.anchor1.to(real_scores)

        if self.measure == 'emd':
            loss = self.distance(self.anchor1, real_scores) + self.distance(self.anchor0, fake_scores)
        else:
            real_probs = softmax(real_scores, dim=1)
            fake_probs = softmax(fake_scores, dim=1)

            loss = self.distance(self.anchor1, real_probs) + self.distance(self.anchor0, fake_probs)
            # loss -= self.div(self.anchor1, fake_probs) + self.div(self.anchor0, real_probs)

        return loss

    def generator_loss(self, real_scores: Tensor, fake_scores: Tensor) -> Tensor:
        self.anchor0 = self.anchor0.to(real_scores)
        self.anchor1 = self.anchor1.to(real_scores)

        if self.measure == 'emd':
            # No relativism
            # loss = self.distance(self.anchor0, fake_probs)

            # EQ19 (default)
            loss = self.distance(real_scores, fake_scores) - self.distance(self.anchor0, fake_scores)

            # EQ20
            # loss = self.distance(self.anchor1, fake_probs) - self.distance(self.anchor0, fake_probs)
        else:
            real_probs = softmax(real_scores, dim=1)
            fake_probs = softmax(fake_scores, dim=1)

            # No relativism
            # loss = self.distance(self.anchor0, fake_probs)

            # EQ19 (default)
            loss = self.distance(real_probs, fake_probs) - self.distance(self.anchor0, fake_probs)

            # EQ20
            # loss = self.distance(self.anchor1, fake_probs) - self.distance(self.anchor0, fake_probs)

        return loss

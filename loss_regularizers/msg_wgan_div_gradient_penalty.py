from typing import List

import torch
from torch import Tensor, nn, autograd


class MsgWganDivGradientPenalty:
    def __init__(self, discriminator: nn.Module, center: float = 0.0, coefficient: float = 2.0, power: int = 6):
        super().__init__()

        self.discriminator = discriminator
        self.center = center
        self.coefficient = coefficient
        self.power = power

    def __call__(self, real_images: List[Tensor], fake_images: List[Tensor], real_scores: Tensor, fake_scores: Tensor):
        batch_size = real_images[0].shape[0]
        device = real_images[0].device

        real_ones = torch.ones_like(real_scores, device=device)
        fake_ones = torch.ones_like(fake_scores, device=device)

        real_gradients = autograd.grad(real_scores, real_images, real_ones, True)
        fake_gradients = autograd.grad(fake_scores, fake_images, fake_ones, True)

        real_gradients = [real_gradient.view(batch_size, -1) for real_gradient in real_gradients]
        fake_gradients = [fake_gradient.view(batch_size, -1) for fake_gradient in fake_gradients]

        real_gradients = torch.cat(real_gradients, dim=1)
        fake_gradients = torch.cat(fake_gradients, dim=1)

        real_gradients_norm = real_gradients.norm(2, dim=1)
        fake_gradients_norm = fake_gradients.norm(2, dim=1)

        real_penalties = (real_gradients_norm - self.center) ** (self.power / 2)
        fake_penalties = (fake_gradients_norm - self.center) ** (self.power / 2)

        return (self.coefficient / 2) * torch.mean(real_penalties + fake_penalties)

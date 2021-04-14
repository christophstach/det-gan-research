import torch
from torch import Tensor, autograd, nn


class GradientPenalty:
    def __init__(self, discriminator: nn.Module) -> None:
        super().__init__()

        self.discriminator = discriminator

    def __call__(self, real_images: Tensor, fake_images: Tensor, real_scores: Tensor, fake_scores: Tensor):
        real_ones = torch.ones_like(real_scores, device=real_images.device)
        fake_ones = torch.ones_like(fake_scores, device=real_images.device)

        real_gradients = autograd.grad(real_scores, real_images, real_ones, True)[0]
        fake_gradients = autograd.grad(fake_scores, fake_images, fake_ones, True)[0]

        real_penalties = real_gradients.norm(2, dim=1) ** 3
        fake_penalties = fake_gradients.norm(2, dim=1) ** 3

        penalties = real_penalties + fake_penalties

        return penalties.mean()

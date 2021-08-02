import torch
from torch import Tensor, autograd, nn


class GradientPenalty:
    def __init__(self, discriminator: nn.Module) -> None:
        super().__init__()

        self.discriminator = discriminator

        self.coefficient = 10.0
        self.center = 0.0
        self.power = 2.0

    def __call__(self, real_images: Tensor, fake_images: Tensor, real_scores: Tensor, fake_scores: Tensor):
        batch_size = real_images.shape[0]
        device = real_images.device

        real_ones = torch.ones_like(real_scores, device=device)
        fake_ones = torch.ones_like(fake_scores, device=device)

        real_gradients = autograd.grad(real_scores, real_images, real_ones, True)[0]
        fake_gradients = autograd.grad(fake_scores, fake_images, fake_ones, True)[0]

        real_gradients = real_gradients.view(batch_size, -1)
        fake_gradients = fake_gradients.view(batch_size, -1)

        real_gradients_norm = real_gradients.norm(2, dim=1)
        fake_gradients_norm = fake_gradients.norm(2, dim=1)

        real_penalties = (real_gradients_norm - self.center) ** (self.power / 2.0)
        fake_penalties = (fake_gradients_norm - self.center) ** (self.power / 2.0)

        return self.coefficient * torch.mean(real_penalties + fake_penalties)

    def __call__2(self, real_images: Tensor, fake_images: Tensor):
        batch_size = real_images.shape[0]

        alpha = torch.rand(batch_size, 1, 1, 1, device=real_images.device, requires_grad=True)

        interpolated_images = real_images + (1 - alpha) * fake_images
        # interpolated_images = real_images + (1 - alpha) * 0.5 * real_images.std() * torch.randn_like(real_images)

        scores = self.discriminator(interpolated_images)
        ones = torch.ones_like(scores, device=real_images.device)

        gradients = autograd.grad(outputs=scores, inputs=interpolated_images, grad_outputs=ones)[0]
        gradients = gradients.view(batch_size, -1)

        penalties = gradients.normZ(2, dim=1) ** 2

        return 10.0 * penalties.mean()

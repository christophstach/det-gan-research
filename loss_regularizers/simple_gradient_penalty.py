import torch
from torch import Tensor, autograd, nn


class GradientPenalty:
    def __init__(self, discriminator: nn.Module) -> None:
        super().__init__()

        self.discriminator = discriminator

    def __call__(self, real_images: Tensor, fake_images: Tensor):
        batch_size = real_images.shape[0]

        alpha = torch.rand(batch_size, 1, 1, 1, device=real_images.device, requires_grad=True)

        # interpolated_images = real_images + (1 - alpha) * fake_images
        interpolated_images = real_images + (1 - alpha) * 0.5 * real_images.std() * torch.randn_like(real_images)

        scores = self.discriminator(interpolated_images)
        ones = torch.ones_like(scores, device=real_images.device)

        gradients = autograd.grad(outputs=scores, inputs=interpolated_images, grad_outputs=ones)[0]
        gradients = gradients.view(batch_size, -1)

        penalties = gradients.norm(2, dim=1) ** 2

        return 10.0 * penalties.mean()

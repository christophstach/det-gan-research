from typing import List

import torch
from determined.pytorch import PyTorchTrialContext

import loss_regularizers.base
import utils


class GradientPenalty(loss_regularizers.base.LossRegularizer):
    def __init__(self,
                 context: PyTorchTrialContext,
                 discriminator: torch.nn.Module,
                 center: float = 0.0,
                 coefficient: float = 10.0,
                 power: int = 2,
                 norm_type: str = "l2",
                 penalty_type: str = "ls",
                 lazy_regularization_interval: int = 16) -> None:

        super().__init__()

        self.context = context
        self.discriminator = discriminator
        self.center = center
        self.coefficient = coefficient
        self.power = power
        self.norm_type = norm_type
        self.penalty_type = penalty_type
        self.lazy_regularization_interval = lazy_regularization_interval
        self.i = 0

    def __call__(self, w: torch.Tensor, real_images: List[torch.Tensor], fake_images: List[torch.Tensor]):
        if self.i % self.lazy_regularization_interval == 0:
            real_images = real_images[-1]
            fake_images = fake_images[-1]

            alpha = torch.rand(real_images.shape[0], 1, 1, 1)
            alpha = self.context.to_device(alpha)

            interpolates = alpha * real_images + (1 - alpha) * fake_images
            interpolates.requires_grad_(True)

            interpolates = utils.to_scaled_images(interpolates, real_images.shape[-1])
            interpolates_validity = self.discriminator(interpolates)

            ones = torch.ones_like(interpolates_validity)
            ones = self.context.to_device(ones)

            inputs_gradients = torch.autograd.grad(
                outputs=interpolates_validity,
                inputs=interpolates,
                grad_outputs=ones,
                create_graph=True
            )

            inputs_gradients = [
                input_gradients.view(input_gradients.shape[0], -1)
                for input_gradients in inputs_gradients
            ]
            gradients = torch.cat(inputs_gradients, dim=1)

            if self.norm_type == "l1":
                gradients_norm = gradients.norm(1, dim=1)
            elif self.norm_type == "l2":
                gradients_norm = gradients.norm(2, dim=1)
            elif self.norm_type == "linf":
                gradients_norm, _ = torch.max(torch.abs(gradients), dim=1)
            else:
                raise NotImplementedError()

            if self.penalty_type == "ls":
                penalties = (gradients_norm - self.center) ** self.power
            elif self.penalty_type == "hinge":
                penalties = torch.relu(gradients_norm - self.center) ** self.power
            else:
                raise NotImplementedError()

            self.i = self.i + 1
            return self.coefficient * penalties.mean().unsqueeze(0)
        else:
            self.i = self.i + 1
            return 0.0

from typing import List

import torch
from determined.pytorch import PyTorchTrialContext
from torch import Tensor

import loss_regularizers.base


class GradientPenalty(loss_regularizers.base.LossRegularizer):
    def __init__(self,
                 context: PyTorchTrialContext,
                 discriminator: torch.nn.Module,
                 center: float = 0.0,
                 coefficient: float = 10.0,
                 power: int = 2,
                 norm_type: str = "l2",
                 penalty_type: str = "ls",
                 lazy_regularization_interval: int = 1) -> None:

        super().__init__()

        self.context = context
        self.discriminator = discriminator
        self.center = center
        self.coefficient = coefficient
        self.power = power
        self.norm_type = norm_type
        self.penalty_type = penalty_type
        self.lazy_regularization_interval = lazy_regularization_interval
        self.steps = 0
        self.last_calculated_gp = None

    def interpolate(self, real_image, fake_image, alpha):
        interpolation = alpha * real_image + (1 - alpha) * fake_image
        interpolation.requires_grad_(True)

        return interpolation

    def __call__(self, w: Tensor, real_images: List[Tensor], fake_images: List[Tensor]):
        if self.coefficient > 0.0 and self.steps % self.lazy_regularization_interval == 0:
            alpha = torch.rand(real_images[0].shape[0], 1, 1, 1)
            alpha = self.context.to_device(alpha)

            interpolations = [
                self.interpolate(real_image, fake_image, alpha)
                for real_image, fake_image
                in zip(real_images, fake_images)
            ]

            interpolations_scores = self.discriminator(interpolations)

            ones = torch.ones_like(interpolations_scores)
            ones = self.context.to_device(ones)

            inputs_gradients = torch.autograd.grad(
                outputs=interpolations_scores,
                inputs=interpolations,
                grad_outputs=ones,
                create_graph=True
            )

            inputs_gradients = [
                gradients.view(gradients.shape[0], -1)
                for gradients in inputs_gradients
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

            gp = self.coefficient * penalties.mean()
            self.last_calculated_gp = gp.clone().detach()
            self.steps += 1

            return gp
        else:
            self.steps += 1
            return self.last_calculated_gp

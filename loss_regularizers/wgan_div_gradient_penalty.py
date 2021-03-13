from typing import List

import torch
from determined.pytorch import PyTorchTrialContext
from torch import Tensor


class WGANDivGradientPenalty:
    def __init__(self,
                 context: PyTorchTrialContext,
                 discriminator: torch.nn.Module,
                 center: float = 0.0,
                 coefficient: float = 2.0,
                 power: int = 6,
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

    def __call__(self, real_images: List[Tensor], fake_images: List[Tensor], real_scores: Tensor, fake_scores: Tensor):
        if self.coefficient > 0.0 and self.steps % self.lazy_regularization_interval == 0:
            real_ones = torch.ones_like(real_scores)
            fake_ones = torch.ones_like(fake_scores)

            real_ones = self.context.to_device(real_ones)
            fake_ones = self.context.to_device(fake_ones)

            real_gradients = torch.autograd.grad(
                outputs=real_scores,
                inputs=real_images,
                grad_outputs=real_ones,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )

            fake_gradients = torch.autograd.grad(
                outputs=fake_scores,
                inputs=fake_images,
                grad_outputs=fake_ones,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )

            real_gradients = [
                real_gradient.view(real_gradient.shape[0], -1)
                for real_gradient in real_gradients
            ]

            fake_gradients = [
                fake_gradient.view(fake_gradient.shape[0], -1)
                for fake_gradient in fake_gradients
            ]

            real_gradients = torch.cat(real_gradients, dim=1)
            fake_gradients = torch.cat(fake_gradients, dim=1)

            if self.norm_type == "l1":
                real_gradients_norm = real_gradients.norm(1, dim=1)
                fake_gradients_norm = fake_gradients.norm(1, dim=1)
            elif self.norm_type == "l2":
                real_gradients_norm = real_gradients.norm(2, dim=1)
                fake_gradients_norm = fake_gradients.norm(2, dim=1)
            elif self.norm_type == "linf":
                real_gradients_norm, _ = torch.max(torch.abs(real_gradients), dim=1)
                fake_gradients_norm, _ = torch.max(torch.abs(fake_gradients), dim=1)
            else:
                raise NotImplementedError()

            if self.penalty_type == "ls":
                real_penalties = (real_gradients_norm - self.center) ** (self.power / 2)
                fake_penalties = (fake_gradients_norm - self.center) ** (self.power / 2)
            elif self.penalty_type == "hinge":
                real_penalties = torch.relu(real_gradients_norm - self.center) ** (self.power / 2)
                fake_penalties = torch.relu(fake_gradients_norm - self.center) ** (self.power / 2)
            else:
                raise NotImplementedError()

            gp = (self.coefficient / 2) * torch.mean(real_penalties + fake_penalties)
            self.last_calculated_gp = gp.clone().detach()
            self.steps += 1

            return gp
        else:
            self.steps += 1
            return self.last_calculated_gp

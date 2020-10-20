import math
from typing import List

import torch
from determined.pytorch import PyTorchTrialContext

import loss_regularizers.base


class PathLengthRegularizer(loss_regularizers.base.LossRegularizer):
    def __init__(self,
                 context: PyTorchTrialContext,
                 coefficient: float = 0.0,
                 decay=0.01,
                 first_step=3000,
                 lazy_regularization_interval: int = 16) -> None:

        super().__init__()

        self.context = context
        self.decay = decay
        self.coefficient = coefficient
        self.moving_mean_path_length = None
        self.lazy_regularization_interval = lazy_regularization_interval
        self.steps = 0
        self.first_step = first_step

    def __call__(self, w: torch.Tensor, real_images: List[torch.Tensor], fake_images: List[torch.Tensor]):
        if self.coefficient > 0.0 and self.steps >= self.first_step and self.steps % self.lazy_regularization_interval == 0:
            fake_images = fake_images[-1]

            noise = torch.randn_like(fake_images) / math.sqrt(
                fake_images.shape[2] * fake_images.shape[3]
            )

            grad = torch.autograd.grad(
                outputs=(fake_images * noise).sum(),
                inputs=w,
                create_graph=True
            )[0]

            path_lengths = torch.sqrt(grad.pow(2).sum(dim=2).mean(dim=1))
            path_lengths_mean = path_lengths.detach().mean()

            if self.moving_mean_path_length is not None:
                self.moving_mean_path_length = self.moving_mean_path_length * self.decay \
                                               + (1 - self.decay) * path_lengths_mean
            else:
                self.moving_mean_path_length = path_lengths_mean

            path_penalty = self.coefficient * ((path_lengths - self.moving_mean_path_length) ** 2).mean()
            self.steps += 1

            return path_penalty
        else:
            self.steps += 1
            return None

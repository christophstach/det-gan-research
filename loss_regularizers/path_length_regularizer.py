import math
from typing import List

import torch
from determined.pytorch import PyTorchTrialContext

import loss_regularizers.base


class PathLengthRegularizer(loss_regularizers.base.LossRegularizer):
    def __init__(self,
                 context: PyTorchTrialContext,
                 weight=1e3,
                 decay=0.01,
                 first_step=3000,
                 lazy_regularization_interval: int = 16) -> None:

        super().__init__()

        self.__context = context
        self.__decay = decay
        self.__weight = weight
        self.__moving_mean_path_length = None
        self.__lazy_regularization_interval = lazy_regularization_interval
        self.__steps = 0
        self.__first_step = first_step

    def __call__(self, w: torch.Tensor, real_images: List[torch.Tensor], fake_images: List[torch.Tensor]):
        if self.__steps >= self.__first_step and self.__steps % self.__lazy_regularization_interval == 0:
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

            if self.__moving_mean_path_length is not None:
                self.__moving_mean_path_length = self.__moving_mean_path_length * self.__decay \
                                                 + (1 - self.__decay) * path_lengths_mean
            else:
                self.__moving_mean_path_length = path_lengths_mean

            path_penalty = self.__weight * ((path_lengths - self.__moving_mean_path_length) ** 2).mean()
            self.__steps += 1

            return path_penalty
        else:
            self.__steps += 1
            return None

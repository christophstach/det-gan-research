import abc

import torch


class Loss(abc.ABC):
    @abc.abstractmethod
    def discriminator_loss(self, real_validity: torch.Tensor, fake_validity: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def generator_loss(self, real_validity: torch.Tensor, fake_validity: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

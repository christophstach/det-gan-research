import abc
from typing import List

import torch


class LossRegularizer(abc.ABC):
    @abc.abstractmethod
    def __call__(self, real_images: List[torch.Tensor], fake_images: List[torch.Tensor]):
        raise NotImplementedError()

from typing import List

import torch
from determined.pytorch import PyTorchTrialContext
from torch import Tensor

import loss_regularizers.base


class OrthogonalRegularizer(loss_regularizers.base.LossRegularizer):
    def __init__(self,
                 context: PyTorchTrialContext,
                 model: torch.nn.Module,
                 coefficient: float = 1e-4,
                 decay: float = 1,
                 lazy_regularization_interval: int = 1) -> None:

        super().__init__()

        self.context = context
        self.model = model
        self.coefficient = coefficient
        self.decay = decay
        self.lazy_regularization_interval = lazy_regularization_interval
        self.steps = 0

    def __call__(self, w: Tensor, real_images: List[Tensor], fake_images: List[Tensor]):
        if self.coefficient > 0.0 and self.steps % self.lazy_regularization_interval == 0:
            with torch.enable_grad():
                orthogonal_loss = torch.zeros(1)
                orthogonal_loss = self.context.to_device(orthogonal_loss)

                for name, param in self.model.named_parameters():
                    if 'bias' not in name:
                        param_flat = param.view(param.shape[0], -1)
                        sym = torch.mm(param_flat, torch.t(param_flat))

                        identity = torch.eye(param_flat.shape[0])
                        identity = self.context.to_device(identity)

                        sym -= identity
                        orthogonal_loss = orthogonal_loss + (self.decay * sym.abs().sum())

            self.steps += 1
            return self.coefficient * orthogonal_loss
        else:
            self.steps += 1
            return None

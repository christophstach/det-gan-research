from typing import List

import torch

import loss_regularizers.base


class OrthogonalRegularizer(loss_regularizers.base.LossRegularizer):
    def __init__(self,
                 model: torch.nn.Module,
                 regularization: float = 1e-6,
                 lazy_regularization_interval: int = 16) -> None:

        super().__init__()

        self.model = model
        self.regularization = regularization
        self.lazy_regularization_interval = lazy_regularization_interval
        self.i = 0

    def __call__(self, w: torch.Tensor, real_images: List[torch.Tensor], fake_images: List[torch.Tensor]):
        if self.i % self.lazy_regularization_interval == 0:
            with torch.enable_grad():
                orthogonal_loss = torch.zeros(1, device=real_images[-1].device)

                for name, param in self.model.named_parameters():
                    if 'bias' not in name:
                        param_flat = param.view(param.shape[0], -1)
                        sym = torch.mm(param_flat, torch.t(param_flat))
                        sym -= torch.eye(param_flat.shape[0], device=real_images[-1].device)
                        orthogonal_loss = orthogonal_loss + (self.regularization * sym.abs().sum())

            self.i = 1
            return orthogonal_loss
        else:
            self.i = self.i + 1
            return 0.0

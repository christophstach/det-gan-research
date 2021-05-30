import torch
from torch import Tensor


class SimplePathLengthRegularizer:
    def __init__(self, coefficient: float = 0.01, decay=0.01) -> None:
        super().__init__()

        self.decay = decay
        self.coefficient = coefficient
        self.mm_path_length = None

    def __call__(self, w: Tensor, fake_images: Tensor):
        y = torch.randn_like(fake_images)

        w_gradients = torch.autograd.grad(
            outputs=(fake_images * y).sum(),
            inputs=w,
            create_graph=True
        )[0]

        # path_lengths = torch.norm(w_gradients)
        path_lengths = torch.sqrt(w_gradients.square().sum(dim=2).mean(dim=1))
        path_lengths_mean = path_lengths.detach().mean()

        if self.mm_path_length is not None:
            self.mm_path_length = self.mm_path_length * self.decay + (1 - self.decay) * path_lengths_mean
        else:
            self.mm_path_length = path_lengths_mean

        path_penalty = self.coefficient * ((path_lengths - self.mm_path_length) ** 2).mean()

        return path_penalty

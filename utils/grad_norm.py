from typing import Iterator

from torch.nn import Parameter


def grad_norm(parameters: Iterator[Parameter]):
    total_norm = 0
    for p in [param for param in parameters if param.grad is not None]:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2

    total_norm = total_norm ** (1. / 2)

    return total_norm

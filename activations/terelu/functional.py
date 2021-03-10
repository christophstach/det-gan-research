"""
https://arxiv.org/abs/2006.02797

"""

# import pytorch
import torch
from torch import Tensor


@torch.jit.script
def terelu(x, beta: Tensor, mu: float, alpha: float):
    c1 = (x <= 0).float()
    c2 = ((mu > x) * (x > 0)).float()
    c3 = (x >= mu).float()

    o1 = c1 * (alpha * (torch.exp(x) - 1))
    o2 = c2 * x
    o3 = c3 * (beta * (mu - (torch.exp(mu - x) - 1)))

    return o1 + o2 + o3

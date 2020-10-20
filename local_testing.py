import torch
from torch.utils.data import DataLoader
from typing import List
import math

import datasets as ds
from models import MsgGenerator, MsgDiscriminator


class PathLengthRegularizer():
    def __init__(self, context=None, decay=0.01) -> None:
        super().__init__()

        self.context = context
        self.decay = decay
        self.moving_mean_path_length = None

    def __call__(self, w: torch.Tensor, real_images: List[torch.Tensor], fake_images: List[torch.Tensor]):
        fake_images = fake_images[-1]

        noise = torch.randn_like(fake_images) / math.sqrt(
            fake_images.shape[2] * fake_images.shape[3]
        )

        grad = torch.autograd.grad(
            outputs=(fake_images * noise).sum(),
            inputs=w,
            create_graph=True,
        )[0]

        # path_lengths = (grad ** 2).sum(dim=1).mean().sqrt()
        path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))
        path_lengths_mean = path_lengths.detach().mean()

        if self.moving_mean_path_length:
            self.moving_mean_path_length = self.moving_mean_path_length * self.decay \
                                           + (1 - self.decay) * path_lengths_mean
        else:
            self.moving_mean_path_length = path_lengths_mean

        path_penalty = ((path_lengths - self.moving_mean_path_length) ** 2).mean()

        return path_penalty


dataset = ds.mnist(True, 32, 3, root=".datasets")
loader = DataLoader(dataset, batch_size=4)

z = torch.rand(4, 128)

generator = MsgGenerator(2, 16, 16, 32, 3, 128, True)
discriminator = MsgDiscriminator(2, 16, 16, 32, 3, True)

imgs, w = generator(z)
plr = PathLengthRegularizer()

print(plr(w, None, imgs))
print(plr(w, None, imgs))
print(plr(w, None, imgs))
print(plr(w, None, imgs))

imgs = torch.randn((3, 3, 32, 32))

print(imgs.shape[-1].item())

import torch
from torch.utils.data import DataLoader
from typing import List
import math

import datasets as ds
from models import MsgGenerator, MsgDiscriminator
from determined.experimental import Checkpoint
from metrics import FrechetInceptionDistance, InceptionScore
from torchvision.models import inception_v3
from torchvision import transforms


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

z = torch.rand(4, 256)

generator = MsgGenerator(
    image_channels=3,
    latent_dimension=256,
    min_filters=0,
    max_filters=256,
    filter_multiplier=4,
    image_size=128,
    spectral_normalization=True
)
discriminator = MsgDiscriminator(
    image_channels=3,
    spectral_normalization=True,
    min_filters=0,
    max_filters=256,
    filter_multiplier=4,
    image_size=128
)

from distributions import TruncatedNormal

dist = TruncatedNormal(0, 1, -2, 2)



max_images_ic = 64
bs = 32

image_stack = None

while True:
    imgs, _ = generator(torch.rand(bs, 256))

    if image_stack is None:
        image_stack = imgs[-1]
    else:
        image_stack = torch.vstack((image_stack, imgs[-1]))

    if image_stack.shape[0] >= max_images_ic:
        break


normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

print(image_stack.sum().item())
print(normalize(image_stack).sum().item())
print(normalize(normalize(image_stack)).sum().item())

ic_model = inception_v3(pretrained=True, aux_logits=False)
ic = InceptionScore(ic_model)
ic.images = image_stack
score = ic()

print(score)
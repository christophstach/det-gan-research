import math

import torch.nn.functional as F
from torch.nn.functional import avg_pool2d
from numpy import log2


def to_scaled_images(source_images, image_size: int, reverse=True):
    images = [source_images] + [
        avg_pool2d(source_images, 2 ** target_size)
        for target_size in range(1, int(log2(image_size)) - 1)
    ]

    if not reverse:
        images = list(reversed(images))

    return images


def to_scaled_images_interpolate(source_images, image_size: int, reverse=True):
    images = [
        *[
            F.interpolate(source_images, size=2 ** target_size)
            for target_size in range(2, int(math.log2(image_size)))
        ],
        source_images
    ]

    if reverse:
        images = list(reversed(images))

    return images

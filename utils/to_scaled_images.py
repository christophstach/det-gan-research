import math

import torch.nn.functional as F
from numpy import log2
from torch.nn.functional import avg_pool2d


def to_scaled_images(source_images, image_size: int, reverse=True, mode='bilinear'):
    if mode == 'avgpool':
        images = [source_images] + [
            avg_pool2d(source_images, 2 ** target_size)
            for target_size in range(1, int(log2(image_size)) - 1)
        ]

        if not reverse:
            images = list(reversed(images))
    elif mode == 'nearest':
        images = [
            *[
                F.interpolate(source_images, size=2 ** target_size, mode='nearest')
                for target_size in range(2, int(math.log2(image_size)))
            ],
            source_images
        ]

        if reverse:
            images = list(reversed(images))
    elif mode == 'bilinear':
        images = [
            *[
                F.interpolate(source_images, size=2 ** target_size, mode='bilinear', align_corners=False)
                for target_size in range(2, int(math.log2(image_size)))
            ],
            source_images
        ]

        if reverse:
            images = list(reversed(images))
    elif mode == 'bicubic':
        images = [
            *[
                F.interpolate(source_images, size=2 ** target_size, mode='bicubic', align_corners=False)
                for target_size in range(2, int(math.log2(image_size)))
            ],
            source_images
        ]

        if reverse:
            images = list(reversed(images))
    else:
        raise ValueError()

    return images

import math

import torch.nn.functional as F


def to_scaled_images(source_images, image_size):
    return [
        *[
            F.interpolate(source_images, size=2 ** target_size)
            for target_size in range(2, int(math.log2(image_size)))
        ],
        source_images
    ]

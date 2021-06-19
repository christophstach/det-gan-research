import math

import torch

from models.hdcgan_discriminator import HdcDiscriminator
from models.hdcgan_generator import HdcGenerator

g_depth = 4
d_depth = 4
latent_dim = 128
score_dim = 32
image_size = 256
image_channels = 3

print('cuda.is_available', torch.cuda.is_available())

z = torch.randn(4, latent_dim).cuda()
generator = HdcGenerator(g_depth, image_size, image_channels, latent_dim).cuda()
discriminator = HdcDiscriminator(d_depth, image_size, image_channels, score_dim).cuda()

print('generator.channels', generator.channels)
print('discriminator.channels', discriminator.channels)

fake_images, _ = generator(z)
print('fake_images.shape', fake_images.shape)

score = discriminator(fake_images)

print(score.shape)

n_blocks = len(generator.blocks)


def calculate_chunk_sections(latent_dim, n_blocks):
    closest_power_of_two = 1
    max_divisible = math.ceil(latent_dim / (n_blocks + 1))

    while True:
        if closest_power_of_two > max_divisible:
            break
        else:
            closest_power_of_two *= 2

    closest_power_of_two //= 2

    sections = [closest_power_of_two for _ in range(n_blocks)]
    sections.insert(0, latent_dim - (n_blocks * closest_power_of_two))

    return sections

import math

import torch
from torch import nn

from layers.pad import EvenPad2d
from layers.res import DownMultiBlock, UpMultiBlock
from models.resgan_discriminator import ResDiscriminator
from models.resgan_generator import ResGenerator

g_depth = 4
d_depth = 4
latent_dim = 128
score_dim = 32
image_size = 256
image_channels = 3

print('cuda.is_available', torch.cuda.is_available())

z = torch.randn(4, latent_dim).cuda()
generator = ResGenerator(g_depth, image_size, image_channels, latent_dim).cuda()
discriminator = ResDiscriminator(d_depth, image_size, image_channels, score_dim).cuda()

print('generator.channels', generator.channels)
print('discriminator.channels', discriminator.channels)

fake_images, w = generator(z)
print('fake_images.shape', fake_images.shape)

score = discriminator(fake_images)

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


def create_chunks(z, n_blocks):
    sections = calculate_chunk_sections(z.shape[1], n_blocks)
    return torch.split(z, sections, dim=1)


for chunk in create_chunks(z, n_blocks):
    print(chunk.shape)

down = DownMultiBlock(image_channels * 4, image_channels * 4).cuda()
up = UpMultiBlock(image_channels * 4, image_channels * 4).cuda()

print(fake_images.shape)
print(down(fake_images.repeat(1, 4, 1, 1)).shape)
print(up(fake_images.repeat(1, 4, 1, 1)).shape)

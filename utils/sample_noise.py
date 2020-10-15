import torch


def sample_noise(batch_size: int, noise_size: int):
    # Could add truncation trick here

    return torch.rand(size=(batch_size, noise_size))

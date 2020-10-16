import torch


def sample_noise(batch_size: int, noise_size: int, normalize=True):
    # Could add truncation trick here
    noise = torch.randn(size=(batch_size, noise_size))

    if normalize:
        return noise / noise.norm(dim=-1, keepdim=True) * (noise_size ** 0.5)
    else:
        return noise

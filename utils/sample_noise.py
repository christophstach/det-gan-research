from torch.distributions import Normal

from distributions import TruncatedNormal

distributions = {
}


def sample_noise(batch_size: int, noise_size: int, normalize=False, truncations=None):
    # Could add truncation trick here

    truncations_key = 'None' if truncations is None else truncations

    if truncations_key not in distributions.keys():
        if truncations_key == 'None':
            distributions[truncations_key] = Normal(0, 1)
        else:
            distributions[truncations_key] = TruncatedNormal(0, 1, truncations[0], truncations[1])

    distribution = distributions[truncations_key]
    noise = distribution.sample((batch_size, noise_size))

    if normalize:
        return noise / noise.norm(dim=-1, keepdim=True) * (noise_size ** 0.5)
    else:
        return noise

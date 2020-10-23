from torch.utils.data import TensorDataset

from utils import sample_noise


def noise(length, noise_size, normalize=False, uniform=False):
    return TensorDataset(
        sample_noise(
            batch_size=length,
            noise_size=noise_size,
            normalize=normalize,
            uniform=uniform
        )
    )

from torch.utils.data import TensorDataset

from utils import sample_noise


def noise(length, noise_size):
    return TensorDataset(
        sample_noise(
            batch_size=length,
            noise_size=noise_size,
        )
    )

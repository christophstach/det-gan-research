import datasets as ds
from torch.utils.data import DataLoader
from metrics import Instability
import torch

noise_ds = ds.noise(16, [2, 2])
loader = DataLoader(noise_ds, shuffle=False, batch_size=4)

stability = Instability()
stability.add_batch(torch.tensor([0.0, 0.0]))
stability.add_batch(torch.tensor([0.0, 1.0]))

stability.step()
stability.add_batch(torch.tensor([2.0, 1.0]))
stability.add_batch(torch.tensor([1.0, 1.0]))


print('Stability: ', stability())

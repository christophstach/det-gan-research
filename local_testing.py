import datasets as ds
from torch.utils.data import DataLoader
from metrics import Instability
import torch

from models import MsgGenerator, MsgDiscriminator
import math

dataset = ds.mnist(True, 32, 3, root=".datasets")
loader = DataLoader(dataset, batch_size=4)

z = torch.rand(4, 128)
generator = MsgGenerator(2, 16, 16, 32, 3, 128, True)
discriminator = MsgDiscriminator(2, 16, 16, 32, 3, True)

imgs = generator(z)
fake_validity = discriminator(imgs)

img_sizes = [
    2 ** (x + 1)
    for x in range(1, int(math.log2(32)))
]


instabilities = [1, 2, 3, 4]


print({

})
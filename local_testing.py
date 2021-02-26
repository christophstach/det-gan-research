import torch

x = torch.rand((32, 3, 32, 32))
x = torch.reshape(x, (-1, x.shape[1] * 2, x.shape[2], x.shape[3]))

print(x.shape)
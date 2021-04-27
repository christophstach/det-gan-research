import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


# https://github.com/stormraiser/gan-weightnorm-resnet/blob/18695b138a86a255f9c38d22be2e80bad1087d6f/modules/TPReLU.py

class TPReLU(Module):

    def __init__(self, num_parameters=1, init=0.25):
        super().__init__()
        self.num_parameters = num_parameters

        self.weight = Parameter(torch.Tensor(num_parameters).fill_(init))
        self.bias = Parameter(torch.zeros(num_parameters))

    def forward(self, x):
        bias_resize = self.bias.view(1, self.num_parameters, *((1,) * (x.dim() - 2))).expand_as(x)
        return F.prelu(x - bias_resize, self.weight.clamp(0, 1)) + bias_resize

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.num_parameters) + ')'

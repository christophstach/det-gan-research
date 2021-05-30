from typing import List

import numpy as np
import torch
from cvxopt import matrix, spmatrix, sparse, solvers
from determined.pytorch import PyTorchTrialContext
from torch import Tensor


class OptimalTransportRegularizer:
    def __init__(self,
                 context: PyTorchTrialContext,
                 discriminator: torch.nn.Module,
                 coefficient: float = 1.0,
                 gamma: int = 0.1) -> None:
        super().__init__()

        self.context = context
        self.discriminator = discriminator

        self.coefficient = coefficient
        self.gamma = gamma
        self.batch_size = 32
        self.data_dim = 3 * 128 * 128

        self.k = 1.0 / self.data_dim if self.coefficient <= 0 else self.coefficient
        self.kr = np.sqrt(self.k)

        self.lamda = 4 * self.kr * self.gamma

        # prepare

        solvers.options['show_progress'] = False
        solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}

        self.A = spmatrix(1.0, range(self.batch_size), [0] * self.batch_size, (self.batch_size, self.batch_size))
        for i in range(1, self.batch_size):
            Ai = spmatrix(1.0, range(self.batch_size), [i] * self.batch_size, (self.batch_size, self.batch_size))
            self.A = sparse([self.A, Ai])

        self.D = spmatrix(-1.0, range(self.batch_size), range(self.batch_size), (self.batch_size, self.batch_size))
        self.DM = self.D
        for i in range(1, self.batch_size):
            self.DM = sparse([self.DM, self.D])

        self.A = sparse([[self.A], [self.DM]])

        cr = matrix([-1.0 / self.batch_size] * self.batch_size)
        cf = matrix([1.0 / self.batch_size] * self.batch_size)
        self.c = matrix([cr, cf])

        self.pStart = {}
        self.pStart['x'] = matrix([matrix([1.0] * self.batch_size), matrix([-1.0] * self.batch_size)])
        self.pStart['s'] = matrix([1.0] * (2 * self.batch_size))

    def wasserstein_lp(self, dist: Tensor, batch_size: int):
        b = matrix(dist.cpu().double().numpy().flatten())
        sol = solvers.lp(self.c, self.A, b, primalstart=self.pStart, solver='glpk')
        offset = 0.5 * (sum(sol['x'])) / self.batch_size
        sol['x'] = sol['x'] - offset
        self.pStart['x'] = sol['x']
        self.pStart['s'] = sol['s']

        return sol

    def __call__(self, fake_images: List[Tensor], fake_scores: Tensor, RF_dif):

        fake_ones = torch.ones_like(fake_scores)
        fake_ones = self.context.to_device(fake_ones)

        inputs_gradients = torch.autograd.grad(
            outputs=fake_scores,
            inputs=fake_images,
            grad_outputs=fake_ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )

        inputs_gradients = [
            gradients.view(gradients.shape[0], -1)
            for gradients in inputs_gradients
        ]
        gradients = torch.cat(inputs_gradients, dim=1)

        penalty = 0.5 * (
            (gradients.norm(dim=1) / (2 * self.kr) - self.kr / 2 * RF_dif.view(gradients.shape[0], -1).norm1(dim=1)).pow(
                2)).mean()

        for fake_image in fake_images:
            fake_image.requires_grad_(False)

            return penalty

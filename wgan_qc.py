from typing import Dict, Any

import numpy as np
import torch
from cvxopt import matrix, spmatrix, sparse, solvers
from determined.pytorch import PyTorchTrial, PyTorchTrialContext, DataLoader, TorchData
from determined.tensorboard.metric_writers.pytorch import TorchWriter
from torch import nn, autograd, Tensor
import utils


class WGANQCTrail(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext) -> None:
        super().__init__(context)

        self.context = context
        self.logger = TorchWriter()

        discriminator_model = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3))
        generator_model = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28))

        self.discriminator = self.context.wrap_model(discriminator_model)
        self.generator = self.context.wrap_model(generator_model)

        self.batch_size = 32
        self.dataset = self.context.get_hparam('dataset')
        self.image_size = self.context.get_hparam('image_size')
        self.image_channels = self.context.get_hparam('image_channels')
        self.latent_dimension = self.context.get_hparam('latent_dimension')

        self.g_optimizer = self.context.get_hparam('g_optimizer')
        self.g_lr = self.context.get_hparam('g_lr')
        self.g_b1 = self.context.get_hparam('g_b1')
        self.g_b2 = self.context.get_hparam('g_b2')

        self.d_optimizer = self.context.get_hparam('d_optimizer')
        self.d_lr = self.context.get_hparam('d_lr')
        self.d_b1 = self.context.get_hparam('d_b1')
        self.d_b2 = self.context.get_hparam('d_b2')

        self.criterion = nn.MSELoss()

        self.opt_d = self.context.wrap_optimizer(
            utils.create_optimizer(
                self.d_optimizer,
                self.discriminator.parameters(),
                self.d_lr,
                (self.d_b1, self.d_b2)
            )
        )

        self.opt_g = self.context.wrap_optimizer(
            utils.create_optimizer(
                self.g_optimizer,
                self.generator.parameters(),
                self.g_lr,
                (self.g_b1, self.g_b2)
            )
        )

        # WGAN-QC
        self.K = -1.0
        self.gamma = 0.1
        self.data_dim = self.batch_size * self.image_channels * self.image_size * self.image_size

        if self.K <= 0:
            self.K = 1.0 / self.data_dim

        self.Kr = np.sqrt(self.K)
        self.LAMBDA = 4 * self.Kr * self.gamma

        # Prepare linear programming solver #
        solvers.options['show_progress'] = False
        solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}

        self.A = spmatrix(1.0, range(self.batch_size), [0] * self.batch_size, (self.batch_size, self.batch_size))

        for i in range(1, self.batch_size):
            self.Ai = spmatrix(1.0, range(self.batch_size), [i] * self.batch_size, (self.batch_size, self.batch_size))
            self.A = sparse([self.A, self.Ai])

        self.D = spmatrix(-1.0, range(self.batch_size), range(self.batch_size), (self.batch_size, self.batch_size))
        self.DM = self.D

        for i in range(1, self.batch_size):
            self.DM = sparse([self.DM, self.D])

        self.A = sparse([[self.A], [self.DM]])

        self.cr = matrix([-1.0 / self.batch_size] * self.batch_size)
        self.cf = matrix([1.0 / self.batch_size] * self.batch_size)
        self.c = matrix([self.cr, self.cf])

        self.pStart = {
            'x': matrix([matrix([1.0] * self.batch_size), matrix([-1.0] * self.batch_size)]),
            's': matrix([1.0] * (2 * self.batch_size))
        }

    def comput_dist(self, real, fake):
        num_r = real.size(0)
        num_f = fake.size(0)
        real_flat = real.view(num_r, -1)
        fake_flat = fake.view(num_f, -1)

        real_3d = real_flat.unsqueeze(1).expand(num_r, num_f, self.data_dim)
        fake_3d = fake_flat.unsqueeze(0).expand(num_r, num_f, self.data_dim)

        # compute squared L2 distance
        dif = real_3d - fake_3d
        dist = 0.5 * dif.pow(2).sum(2).squeeze()

        return dist

    def Wasserstein_LP(self, dist):
        b = matrix(dist.cpu().double().numpy().flatten())
        sol = solvers.lp(self.c, self.A, b, primalstart=self.pStart, solver='glpk')
        offset = 0.5 * (sum(sol['x'])) / self.batch_size
        sol['x'] = sol['x'] - offset
        self.pStart['x'] = sol['x']
        self.pStart['s'] = sol['s']

        return sol

    def approx_OT(self, sol):
        # ResMat = np.array(sol['s']).reshape((batchSize,batchSize))
        # mapping = torch.from_numpy(np.argmin(ResMat, axis=0)).long().to(device)

        ResMat = np.array(sol['z']).reshape((self.batch_size, self.batch_size))

        mapping = torch.from_numpy(np.argmax(ResMat, axis=0)).long()
        mapping = self.context.to_device(mapping)

        return mapping

    def OT_regularization(self, output_fake: Tensor, fake: Tensor, RF_dif):
        output_fake_grad = torch.ones(output_fake.size())
        output_fake_grad = self.context.to_device(output_fake_grad)

        gradients = autograd.grad(
            outputs=output_fake, inputs=fake,
            grad_outputs=output_fake_grad,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        n = gradients.shape[0]

        RegLoss = 0.5 * (
            (gradients.view(n, -1).norm(dim=1) / (2 * self.Kr) - self.Kr / 2 * RF_dif.view(n, -1).norm(dim=1)).pow(2)
        ).mean()

        fake.requires_grad_(False)

        return RegLoss

    def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int) -> Dict[str, Any]:
        real_images, _ = batch

        # discriminator
        z = utils.sample_noise(self.batch_size, self.latent_dimension)
        z = self.context.to_device(z)

        with torch.no_grad():
            fake_images = self.generator(z)

        dist = self.K * self.comput_dist(real_images, fake_images)
        sol = self.Wasserstein_LP(dist)

        if self.LAMBDA > 0:
            mapping = self.approx_OT(sol)
            real_images_ordered = real_images[mapping]  # match real and fake
            RF_dif = real_images_ordered - fake_images

        # construct target
        target = torch.from_numpy(np.array(sol['x'])).float()
        target = target.squeeze()
        target = self.context.to_device(target)

        self.discriminator.zero_grad()
        fake_images.requires_grad_()
        if fake_images.grad is not None:
            fake_images.grad.data.zero_()
        output_real = self.discriminator(real_images)
        output_fake = self.discriminator(fake_images)
        output_real, output_fake = output_real.squeeze(), output_fake.squeeze()
        output_R_mean = output_real.mean(0).view(1)
        output_F_mean = output_fake.mean(0).view(1)

        L2LossD_real = self.criterion(output_R_mean[0], target[:self.batch_size].mean())
        L2LossD_fake = self.criterion(output_fake, target[self.batch_size:])
        L2LossD = 0.5 * L2LossD_real + 0.5 * L2LossD_fake

        if self.LAMBDA > 0:
            RegLossD = self.OT_regularization(output_fake, fake_images, RF_dif)
            TotalLoss = L2LossD + self.LAMBDA * RegLossD
        else:
            TotalLoss = L2LossD

        self.context.backward(TotalLoss)
        self.context.step_optimizer(self.opt_d)

        WD = output_R_mean - output_F_mean  # Wasserstein Distance

        return {}

    def build_training_data_loader(self) -> DataLoader:
        pass

    def build_validation_data_loader(self) -> DataLoader:
        pass

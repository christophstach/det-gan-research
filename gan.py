"""
This example demonstrates how to train a GAN with Determined's PyTorch API.

The PyTorch API supports multiple model graphs, optimizers, and LR
schedulers. Those objects should be created and wrapped in the trial class's
__init__ method. Then in train_batch(), you can run forward and backward passes
and step the optimizer according to your requirements.
"""

from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from determined.pytorch import PyTorchTrial, PyTorchTrialContext, DataLoader
from determined.tensorboard.metric_writers.pytorch import TorchWriter
from torchvision.utils import make_grid
import torch.optim as optim

import datasets as ds
from metrics import Instability
from utils.types import TorchData


class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]

            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))

            layers.append(nn.LeakyReLU(0.2, inplace=True))

            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


class GANTrial(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext) -> None:
        super().__init__(context)
        self.context = context
        self.logger = TorchWriter()

        self.latent_dim = self.context.get_hparam("latent_dim")

        # Initialize the models.
        mnist_shape = (1, 32, 32)
        self.generator = self.context.wrap_model(
            Generator(
                latent_dim=self.latent_dim,
                img_shape=mnist_shape
            )
        )

        self.discriminator = self.context.wrap_model(Discriminator(img_shape=mnist_shape))

        # Initialize the optimizers and learning rate scheduler.
        lr = self.context.get_hparam("lr")
        b1 = self.context.get_hparam("b1")
        b2 = self.context.get_hparam("b2")

        self.opt_g = self.context.wrap_optimizer(
            optim.Adam(
                self.generator.parameters(),
                lr=lr,
                betas=(b1, b2)
            )
        )

        # self.opt_g = self.context.wrap_optimizer(
        #     Lamb(
        #         self.generator.parameters(),
        #         lr=lr,
        #         betas=(b1, b2)
        #     )
        # )

        self.opt_d = self.context.wrap_optimizer(
            optim.Adam(
                self.discriminator.parameters(),
                lr=lr,
                betas=(b1, b2)
            )
        )

        # self.opt_d = self.context.wrap_optimizer(
        #    Lamb(
        #        self.generator.parameters(),
        #        lr=lr,
        #        betas=(b1, b2)
        #    )
        # )

        self.instability_metric = Instability()

    def build_training_data_loader(self) -> DataLoader:
        train_data = ds.mnist(train=True, size=32)
        return DataLoader(
            train_data,
            batch_size=self.context.get_per_slot_batch_size()
        )

    def build_validation_data_loader(self) -> DataLoader:
        validation_data = ds.noise(self.context.get_per_slot_batch_size(), [self.latent_dim])

        return DataLoader(
            validation_data,
            batch_size=self.context.get_per_slot_batch_size(),
            shuffle=False
        )

    def sample_noise(self, batch_size):
        return torch.randn(batch_size, self.context.get_hparam("latent_dim"))

    def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int) -> Dict[str, torch.Tensor]:
        imgs, _ = batch

        # Train generator.
        # Set `requires_grad_` to only update parameters on the generator.
        self.generator.requires_grad_(True)
        self.discriminator.requires_grad_(False)

        # Sample noise and generator images.
        # Note that you need to map the generated data to the device specified by Determined.
        z = self.sample_noise(imgs.shape[0])
        z = self.context.to_device(z)
        generated_imgs = self.generator(z)

        # Calculate generator loss.
        valid = torch.ones(imgs.size(0), 1)
        valid = self.context.to_device(valid)
        g_loss = F.binary_cross_entropy(self.discriminator(generated_imgs), valid)

        # Run backward pass and step the optimizer for the generator.
        self.context.backward(g_loss)
        self.context.step_optimizer(self.opt_g)

        # Train discriminator.
        # Set `requires_grad_` to only update parameters on the discriminator.
        self.generator.requires_grad_(False)
        self.discriminator.requires_grad_(True)

        # Calculate discriminator loss with a batch of real images and a batch of fake images.
        valid = torch.ones(imgs.size(0), 1)
        valid = self.context.to_device(valid)
        real_loss = F.binary_cross_entropy(self.discriminator(imgs), valid)
        fake = torch.zeros(generated_imgs.size(0), 1)
        fake = self.context.to_device(fake)
        fake_loss = F.binary_cross_entropy(self.discriminator(generated_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        # Run backward pass and step the optimizer for the generator.
        self.context.backward(d_loss)
        self.context.step_optimizer(self.opt_d)

        return {
            'loss': d_loss,
            'g_loss': g_loss,
            'd_loss': d_loss,
        }

    def evaluate_full_dataset(self, data_loader: DataLoader) -> Dict[str, Any]:
        generated_fixed_imgs = None

        for [z1] in data_loader:
            z1 = self.context.to_device(z1)
            generated_fixed_imgs = self.generator(z1)
            self.instability_metric.add_batch(generated_fixed_imgs)

        instability = self.instability_metric()
        self.instability_metric.step()

        # Log fix images to Tensorboard.
        sample_fixed_imgs = generated_fixed_imgs[:6]
        grid = make_grid(sample_fixed_imgs)
        self.logger.writer.add_image(f'generated_fixed_images', grid)

        # Log sample images to Tensorboard.
        z2 = self.sample_noise(6)
        z2 = self.context.to_device(z2)
        generated_sample_imgs = self.generator(z2)
        grid = make_grid(generated_sample_imgs)
        self.logger.writer.add_image(f'generated_sample_images', grid)

        return {
            'instability': instability
        }

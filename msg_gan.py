import math
from typing import Any, Dict

import torch
import torch.optim as optim
from determined.pytorch import PyTorchTrial, PyTorchTrialContext, DataLoader
from determined.tensorboard.metric_writers.pytorch import TorchWriter
from torchvision.utils import make_grid

import datasets as ds
import utils
from loss_regularizers import GradientPenalty
from losses import WGAN
from metrics import Instability
from models import MsgDiscriminator, MsgGenerator
from utils.types import TorchData


class MsgGANTrail(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext) -> None:
        super().__init__(context)

        self.context = context
        self.logger = TorchWriter()

        lr = self.context.get_hparam("lr")
        b1 = self.context.get_hparam("b1")
        b2 = self.context.get_hparam("b2")

        filter_multiplier = self.context.get_hparam("filter_multiplier")
        min_filters = self.context.get_hparam("min_filters")
        max_filters = self.context.get_hparam("max_filters")
        spectral_normalization = self.context.get_hparam("spectral_normalization")

        self.image_size = self.context.get_hparam("image_size")
        self.image_channels = self.context.get_hparam("image_channels")
        self.latent_dimension = self.context.get_hparam("latent_dimension")

        self.generator = self.context.wrap_model(
            MsgGenerator(
                filter_multiplier=filter_multiplier,
                min_filters=min_filters,
                max_filters=max_filters,
                image_size=self.image_size,
                image_channels=self.image_channels,
                latent_dimension=self.latent_dimension,
                spectral_normalization=spectral_normalization
            )
        )

        self.discriminator = self.context.wrap_model(
            MsgDiscriminator(
                filter_multiplier=filter_multiplier,
                min_filters=min_filters,
                max_filters=max_filters,
                image_size=self.image_size,
                image_channels=self.image_channels,
                spectral_normalization=spectral_normalization
            )
        )

        self.opt_g = self.context.wrap_optimizer(optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2)))
        self.opt_d = self.context.wrap_optimizer(optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2)))

        self.loss = WGAN()
        self.gradient_penalty = GradientPenalty(self.context, self.discriminator)

        self.img_sizes = [
            2 ** (x + 1)
            for x in range(1, int(math.log2(self.image_size)))
        ]

        self.instability_metrics = [Instability() for _ in self.img_sizes]

    def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int) -> Dict[str, torch.Tensor]:
        real_imgs, _ = batch
        scaled_real_images = utils.to_scaled_images(real_imgs, self.image_size)

        # Train generator.
        # Set `requires_grad_` to only update parameters on the generator.
        self.generator.requires_grad_(True)
        self.discriminator.requires_grad_(False)

        # Sample noise and generator images.
        # Note that you need to map the generated data to the device specified by Determined.
        z = utils.sample_noise(real_imgs.shape[0], self.latent_dimension)
        z = self.context.to_device(z)
        scaled_fake_imgs = self.generator(z)

        real_validity = self.discriminator(scaled_real_images)
        fake_validity = self.discriminator(scaled_fake_imgs)

        # Calculate generator loss.
        g_loss = self.loss.generator_loss(real_validity, fake_validity)

        # Run backward pass and step the optimizer for the generator.
        self.context.backward(g_loss)
        self.context.step_optimizer(self.opt_g)

        # Train discriminator.
        # Set `requires_grad_` to only update parameters on the discriminator.
        self.generator.requires_grad_(False)
        self.discriminator.requires_grad_(True)

        # Calculate discriminator loss with a batch of real images and a batch of fake images.
        real_validity = self.discriminator(scaled_real_images)
        fake_validity = self.discriminator([img.detach() for img in scaled_fake_imgs])
        gp = self.gradient_penalty(scaled_real_images, scaled_fake_imgs)
        d_loss = self.loss.discriminator_loss(real_validity, fake_validity)

        # Run backward pass and step the optimizer for the generator.
        self.context.backward(d_loss)
        self.context.step_optimizer(self.opt_d)

        return {
            "loss": d_loss + gp,
            "g_loss": g_loss,
            "d_loss": d_loss,
            "gp": gp
        }

    def build_training_data_loader(self) -> DataLoader:
        train_data = ds.mnist(train=True, size=self.image_size, channels=self.image_channels)

        return DataLoader(
            train_data,
            batch_size=self.context.get_per_slot_batch_size(),
            drop_last=True
        )

    def build_validation_data_loader(self) -> DataLoader:
        validation_data = ds.noise(self.context.get_per_slot_batch_size(), [self.latent_dimension])

        return DataLoader(
            validation_data,
            batch_size=self.context.get_per_slot_batch_size(),
            shuffle=False
        )

    def evaluate_full_dataset(self, data_loader: DataLoader) -> Dict[str, Any]:
        generated_fixed_imgs = None

        for [z1] in data_loader:
            z1 = self.context.to_device(z1)
            generated_fixed_imgs = self.generator(z1)

            for generated_fixed_img, instability_metric in zip(generated_fixed_imgs, self.instability_metrics):
                instability_metric.add_batch(generated_fixed_img)

        instabilities = [instability_metric() for instability_metric in self.instability_metrics]

        for instability_metric in self.instability_metrics:
            instability_metric.step()

        # Log fix images to Tensorboard.
        sample_fixed_imgs = generated_fixed_imgs[-1][:6]
        grid = make_grid(sample_fixed_imgs)
        self.logger.writer.add_image(f'generated_fixed_images', grid)

        # Log sample images to Tensorboard.
        z2 = utils.sample_noise(6, self.latent_dimension)
        z2 = self.context.to_device(z2)
        generated_sample_imgs = self.generator(z2)
        grid = make_grid(generated_sample_imgs[-1])
        self.logger.writer.add_image(f'generated_sample_images', grid)

        return {
            **{
                f'{size}x{size}_instability': instability
                for size, instability
                in zip(self.img_sizes, instabilities)
            }
        }

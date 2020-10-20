import math
from typing import Any, Dict

import torch
import torch.optim as optim
from determined.pytorch import PyTorchTrial, PyTorchTrialContext, DataLoader
from determined.tensorboard.metric_writers.pytorch import TorchWriter
from torchvision.utils import make_grid

import datasets as ds
import utils
from loss_regularizers import GradientPenalty, PathLengthRegularizer
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
        self.num_log_images = 6
        self.log_images_interval = 1000

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
        self.path_length_regularizer = PathLengthRegularizer(self.context)

        self.img_sizes = [
            2 ** (x + 1)
            for x in range(1, int(math.log2(self.image_size)))
        ]

        self.fixed_z = None
        self.instability_metrics = [Instability() for _ in self.img_sizes]

    def optimize_discriminator(self, z, scaled_real_images):
        self.generator.requires_grad_(False)
        self.discriminator.requires_grad_(True)

        scaled_fake_images, w = self.generator(z)

        real_validity = self.discriminator(scaled_real_images)
        fake_validity = self.discriminator([img.detach() for img in scaled_fake_images])

        gp = self.gradient_penalty(w, scaled_real_images, scaled_fake_images)
        d_loss = self.loss.discriminator_loss(real_validity, fake_validity)

        if gp is not None:
            self.context.backward(d_loss + gp)
        else:
            self.context.backward(d_loss)

        self.context.step_optimizer(self.opt_d)

        return d_loss, gp

    def optimize_generator(self, z, scaled_real_images):
        self.generator.requires_grad_(True)
        self.discriminator.requires_grad_(False)

        scaled_fake_images, w = self.generator(z)
        real_validity = self.discriminator(scaled_real_images)
        fake_validity = self.discriminator(scaled_fake_images)

        plr = self.path_length_regularizer(w, scaled_real_images, scaled_fake_images)
        g_loss = self.loss.generator_loss(real_validity, fake_validity)

        if plr is not None:
            self.context.backward(g_loss + plr)
        else:
            self.context.backward(g_loss)

        self.context.step_optimizer(self.opt_g)

        return g_loss, plr

    def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int) -> Dict[str, torch.Tensor]:
        real_imgs, _ = batch
        scaled_real_images = utils.to_scaled_images(real_imgs, self.image_size)
        z = utils.sample_noise(real_imgs.shape[0], self.latent_dimension)
        z = self.context.to_device(z)

        d_loss, gp = self.optimize_discriminator(z, scaled_real_images)
        g_loss, plr = self.optimize_generator(z, scaled_real_images)

        if batch_idx % self.log_images_interval == 0:
            self.log_fixed_images(batch_idx, epoch_idx)
            self.log_sample_images(batch_idx, epoch_idx)

        logs = {
            "loss": d_loss,
            "g_loss": g_loss,
            "d_loss": d_loss,
            "gp": gp,
            "plr": plr
        }

        return {
            key: (logs[key] if logs[key] is not None else 0.0)
            for key in logs
        }

    def build_training_data_loader(self) -> DataLoader:
        train_data = ds.celeba_hq(size=self.image_size, channels=self.image_channels)

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

    def log_fixed_images(self, batch_idx, epoch_idx):
        if self.fixed_z is None:
            self.fixed_z = utils.sample_noise(self.num_log_images, self.latent_dimension)
            self.fixed_z = self.context.to_device(self.fixed_z)

        fixed_imgs, _ = self.generator(self.fixed_z)
        for imgs in fixed_imgs:
            size = str(imgs.shape[-1])
            grid = make_grid(imgs)
            self.logger.writer.add_image(f'generated_fixed_images_{size}x{size}_epoch_{epoch_idx}', grid, batch_idx)

    def log_sample_images(self, batch_idx, epoch_idx):
        z = utils.sample_noise(self.num_log_images, self.latent_dimension)
        z = self.context.to_device(z)

        sample_imgs, _ = self.generator(z)
        for imgs in sample_imgs:
            size = str(imgs.shape[-1])
            grid = make_grid(imgs)
            self.logger.writer.add_image(f'generated_sample_images_{size}x{size}_epoch_{epoch_idx}', grid, batch_idx)

    def evaluate_full_dataset(self, data_loader: DataLoader) -> Dict[str, Any]:
        for [z1] in data_loader:
            z1 = self.context.to_device(z1)
            generated_fixed_imgs, _ = self.generator(z1)

            for generated_fixed_img, instability_metric in zip(generated_fixed_imgs, self.instability_metrics):
                instability_metric.add_batch(generated_fixed_img)

        instabilities = [instability_metric() for instability_metric in self.instability_metrics]

        for instability_metric in self.instability_metrics:
            instability_metric.step()

        return {
            **{
                f'{size}x{size}_instability': instability
                for size, instability
                in zip(self.img_sizes, instabilities)
            }
        }

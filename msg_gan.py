from typing import Union, Any, Dict

import torch
import torch.nn.functional as F
import torch.optim as optim
from determined.pytorch import PyTorchTrial, PyTorchTrialContext, DataLoader
from determined.tensorboard.metric_writers.pytorch import TorchWriter

import data
from losses import WGAN
from models import MsgDiscriminator, MsgGenerator

from .utils.types import TorchData


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

    def train_batch(
            self,
            batch: TorchData,
            epoch_idx: int,
            batch_idx: int,
            **kwargs
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        imgs, _ = batch

        # Train generator.
        # Set `requires_grad_` to only update parameters on the generator.
        self.generator.requires_grad_(True)
        self.discriminator.requires_grad_(False)

        # Sample noise and generator images.
        # Note that you need to map the generated data to the device specified by Determined.
        z = torch.randn(imgs.shape[0], self.latent_dimension)
        z = self.context.to_device(z)
        generated_imgs = self.generator(z)

        # Log sampled images to Tensorboard.
        # sample_imgs = generated_imgs[:6]
        # grid = make_grid(sample_imgs)
        # self.logger.writer.add_image(f"generated_images_epoch_{epoch_idx}", grid, batch_idx)

        real_validity = self.discriminator(imgs)
        fake_validity = self.discriminator(generated_imgs)

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
        real_validity = self.discriminator(imgs)
        fake_validity = self.discriminator(generated_imgs.detach())
        d_loss = self.loss.discriminator_loss(real_validity, fake_validity)

        # Run backward pass and step the optimizer for the generator.
        self.context.backward(d_loss)
        self.context.step_optimizer(self.opt_d)

        return {
            "loss": d_loss,
            "g_loss": g_loss,
            "d_loss": d_loss,
        }

    def build_training_data_loader(self) -> DataLoader:
        if not self.data_downloaded:
            self.download_directory = data.download_dataset(
                download_directory=self.download_directory,
                data_config=self.context.get_data_config(),
            )
            self.data_downloaded = True

        train_data = data.get_dataset(self.download_directory, train=True)

        return DataLoader(train_data, batch_size=self.context.get_per_slot_batch_size())

    def build_validation_data_loader(self) -> DataLoader:
        if not self.data_downloaded:
            self.download_directory = data.download_dataset(
                download_directory=self.download_directory,
                data_config=self.context.get_data_config(),
            )
            self.data_downloaded = True

        validation_data = data.get_dataset(self.download_directory, train=False)
        return DataLoader(validation_data, batch_size=self.context.get_per_slot_batch_size())

    def evaluate_batch(self, batch: TorchData) -> Dict[str, Any]:
        imgs, _ = batch
        valid = torch.ones(imgs.size(0), 1)
        valid = self.context.to_device(valid)
        loss = F.binary_cross_entropy(self.discriminator(imgs), valid)

        return {"loss": loss}

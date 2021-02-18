from typing import Any, Dict

import math
import torch
from determined.pytorch import PyTorchTrial, PyTorchTrialContext, DataLoader
from determined.tensorboard.metric_writers.pytorch import TorchWriter
from torchvision.utils import make_grid
from torch.nn.utils import clip_grad_norm_

import datasets as ds
import utils
from loss_regularizers import GradientPenalty, PathLengthRegularizer, OrthogonalRegularizer
from metrics import Instability, InceptionScore
from models import MsgDiscriminator, MsgGenerator, ExponentialMovingAverage
from utils.types import TorchData


class MsgGANTrail(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext) -> None:
        super().__init__(context)

        self.context = context
        self.logger = TorchWriter()

        self.dataset = self.context.get_hparam("dataset")
        self.image_size = self.context.get_hparam("image_size")
        self.image_channels = self.context.get_hparam("image_channels")

        self.loss_fn = self.context.get_hparam("loss_fn")
        self.latent_dimension = self.context.get_hparam("latent_dimension")
        self.ema = self.context.get_hparam("ema")
        self.ema_decay = self.context.get_hparam("ema_decay")
        self.instance_noise_until = self.context.get_hparam("instance_noise_until")
        self.clip_grad_norm = self.context.get_hparam("clip_grad_norm")

        self.gradient_penalty_coefficient = self.context.get_hparam("gradient_penalty_coefficient")
        self.path_length_regularizer_coefficient = self.context.get_hparam("path_length_regularizer_coefficient")
        self.d_orthogonal_regularizer_coefficient = self.context.get_hparam("d_orthogonal_regularizer_coefficient")
        self.g_orthogonal_regularizer_coefficient = self.context.get_hparam("g_orthogonal_regularizer_coefficient")

        self.g_optimizer = self.context.get_hparam("g_optimizer")
        self.g_lr = self.context.get_hparam("g_lr")
        self.g_b1 = self.context.get_hparam("g_b1")
        self.g_b2 = self.context.get_hparam("g_b2")

        self.d_optimizer = self.context.get_hparam("d_optimizer")
        self.d_lr = self.context.get_hparam("d_lr")        
        self.d_b1 = self.context.get_hparam("d_b1")
        self.d_b2 = self.context.get_hparam("d_b2")

        self.g_depth = self.context.get_hparam("g_depth")
        self.g_min_filters = self.context.get_hparam("g_min_filters")
        self.g_max_filters = self.context.get_hparam("g_max_filters")

        self.d_depth = self.context.get_hparam("d_depth")
        self.d_min_filters = self.context.get_hparam("d_min_filters")
        self.d_max_filters = self.context.get_hparam("d_max_filters")

        self.g_activation_fn = self.context.get_hparam("g_activation_fn")
        self.g_normalization = self.context.get_hparam("g_normalization")
        self.g_spectral_normalization = self.context.get_hparam("g_spectral_normalization")

        self.d_activation_fn = self.context.get_hparam("d_activation_fn")
        self.d_normalization = self.context.get_hparam("d_normalization")
        self.d_spectral_normalization = self.context.get_hparam("d_spectral_normalization")

        self.inception_score_images = self.context.get_hparam("inception_score_images")
        self.evaluation_model = self.context.get_hparam("evaluation_model")
        self.num_log_images = 6
        self.log_images_interval = 1000

        generator_model = MsgGenerator(
            depth=self.g_depth,
            min_filters=self.g_min_filters,
            max_filters=self.g_max_filters,
            image_size=self.image_size,
            image_channels=self.image_channels,
            latent_dimension=self.latent_dimension,
            activation_fn=self.g_activation_fn,
            normalization=self.g_normalization,
            spectral_normalization=self.g_spectral_normalization
        )

        discriminator_model = MsgDiscriminator(
            depth=self.d_depth,
            min_filters=self.d_min_filters,
            max_filters=self.d_max_filters,
            image_size=self.image_size,
            image_channels=self.image_channels,
            activation_fn=self.d_activation_fn,
            normalization=self.d_normalization,
            spectral_normalization=self.d_spectral_normalization
        )

        generator_model = ExponentialMovingAverage(generator_model, self.ema_decay) if self.ema else generator_model

        self.generator = self.context.wrap_model(generator_model)
        self.discriminator = self.context.wrap_model(discriminator_model)

        self.opt_g = self.context.wrap_optimizer(
            utils.create_optimizer(
                self.g_optimizer,
                self.generator.parameters(),
                self.g_lr,
                (self.g_b1, self.g_b2)
            )
        )

        self.opt_d = self.context.wrap_optimizer(
            utils.create_optimizer(
                self.d_optimizer,
                self.discriminator.parameters(),
                self.d_lr,
                (self.d_b1, self.d_b2)
            )
        )

        self.loss = utils.create_loss_fn(self.loss_fn)

        self.gradient_penalty = GradientPenalty(
            self.context,
            self.discriminator,
            coefficient=self.gradient_penalty_coefficient
        )

        self.path_length_regularizer = PathLengthRegularizer(
            self.context,
            coefficient=self.path_length_regularizer_coefficient
        )

        self.g_orthogonal_regularizer = OrthogonalRegularizer(
            self.context,
            self.generator,
            self.g_orthogonal_regularizer_coefficient
        )

        self.d_orthogonal_regularizer = OrthogonalRegularizer(
            self.context,
            self.discriminator,
            self.d_orthogonal_regularizer_coefficient
        )

        self.img_sizes = [
            2 ** (x + 1)
            for x in range(1, int(math.log2(self.image_size)))
        ]

        eval_model, resize_to, num_classes = utils.create_evaluation_model(self.evaluation_model)
        eval_model = self.context.wrap_model(eval_model)

        self.fixed_z = None
        self.instability_metrices = [Instability() for _ in self.img_sizes]
        self.inception_score_metric = InceptionScore(
            eval_model,
            resize_to,
            num_classes,
            self.context.get_per_slot_batch_size()
        )

    def optimize_discriminator(self, z, scaled_real_images, batch_idx):
        self.generator.requires_grad_(False)
        self.discriminator.requires_grad_(True)

        scaled_fake_images, w = self.generator(z)

        scaled_real_images, in_sigma = utils.instance_noise(scaled_real_images, batch_idx, self.instance_noise_until)
        scaled_fake_images, in_sigma = utils.instance_noise(scaled_fake_images, batch_idx, self.instance_noise_until)

        real_validity = self.discriminator(scaled_real_images)
        fake_validity = self.discriminator([img.detach() for img in scaled_fake_images])

        gp = self.gradient_penalty(w, scaled_real_images, scaled_fake_images)
        d_ortho = self.d_orthogonal_regularizer(w, scaled_real_images, scaled_fake_images)
        d_loss = self.loss.discriminator_loss(real_validity, fake_validity)

        total = d_loss
        total = total + gp if gp is not None else total
        total = total + d_ortho if d_ortho is not None else total

        self.context.backward(total)
        if self.clip_grad_norm > 0: clip_grad_norm_(self.discriminator.parameters(), self.clip_grad_norm)
        d_grad_norm = utils.grad_norm(self.discriminator.parameters())
        self.context.step_optimizer(self.opt_d)

        return d_loss, gp, d_ortho, d_grad_norm, in_sigma

    def optimize_generator(self, z, scaled_real_images, batch_idx):
        self.generator.requires_grad_(True)
        self.discriminator.requires_grad_(False)

        scaled_fake_images, w = self.generator(z)

        scaled_real_images, in_sigma = utils.instance_noise(scaled_real_images, batch_idx, self.instance_noise_until)
        scaled_fake_images, in_sigma = utils.instance_noise(scaled_fake_images, batch_idx, self.instance_noise_until)

        real_validity = self.discriminator(scaled_real_images)
        fake_validity = self.discriminator(scaled_fake_images)

        plr = self.path_length_regularizer(w, scaled_real_images, scaled_fake_images)
        g_ortho = self.g_orthogonal_regularizer(w, scaled_real_images, scaled_fake_images)
        g_loss = self.loss.generator_loss(real_validity, fake_validity)

        total = g_loss
        total = total + plr if plr is not None else total
        total = total + g_ortho if g_ortho is not None else total

        self.context.backward(total)
        if self.clip_grad_norm > 0: clip_grad_norm_(self.generator.parameters(), self.clip_grad_norm)
        g_grad_norm = utils.grad_norm(self.generator.parameters())
        self.context.step_optimizer(self.opt_g)

        if self.ema:
            self.generator.update()

        return g_loss, plr, g_ortho, g_grad_norm, in_sigma

    def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int) -> Dict[str, torch.Tensor]:
        real_imgs, _ = batch
        scaled_real_images = utils.to_scaled_images(real_imgs, self.image_size)
        z = utils.sample_noise(real_imgs.shape[0], self.latent_dimension)
        z = self.context.to_device(z)

        d_loss, gp, d_ortho, d_grad_norm, in_sigma = self.optimize_discriminator(z, scaled_real_images, batch_idx)
        g_loss, plr, g_ortho, g_grad_norm, in_sigma = self.optimize_generator(z, scaled_real_images, batch_idx)

        if batch_idx % self.log_images_interval == 0:
            self.log_fixed_images(batch_idx, epoch_idx)
            self.log_sample_images(batch_idx, epoch_idx)

        logs = {
            "loss": d_loss + (gp if gp else 0.0),
            "g_loss": g_loss,
            "d_loss": d_loss,
            "gp": gp,
            "plr": plr,
            "g_ortho": g_ortho,
            "d_ortho": d_ortho,
            "in_sigma": in_sigma,
            "g_grad_norm": g_grad_norm,
            "d_grad_norm": d_grad_norm
        }

        return {
            key: (logs[key] if logs[key] is not None else 0.0)
            for key in logs
        }

    def build_training_data_loader(self) -> DataLoader:
        train_data = utils.create_dataset(dataset=self.dataset, size=self.image_size, channels=self.image_channels)

        return DataLoader(
            train_data,
            batch_size=self.context.get_per_slot_batch_size(),
            drop_last=True
        )

    def build_validation_data_loader(self) -> DataLoader:
        validation_data = ds.noise(
            length=self.context.get_per_slot_batch_size(),
            noise_size=self.latent_dimension
        )

        return DataLoader(
            validation_data,
            batch_size=self.context.get_per_slot_batch_size(),
            shuffle=False
        )

    def log_fixed_images(self, batch_idx, epoch_idx):
        self.generator.eval()

        if self.fixed_z is None:
            self.fixed_z = utils.sample_noise(
                self.num_log_images,
                self.latent_dimension
            )
            self.fixed_z = self.context.to_device(self.fixed_z)

        fixed_imgs, _ = self.generator(self.fixed_z)
        for imgs in fixed_imgs:
            size = str(imgs.shape[-1])
            imgs = utils.shift_image_range(imgs)
            grid = make_grid(imgs)
            self.logger.writer.add_image(f'generated_fixed_images_{size}x{size}', grid, batch_idx)

        self.generator.train()

    def log_sample_images(self, batch_idx, epoch_idx):
        self.generator.eval()

        z = utils.sample_noise(
            self.num_log_images,
            self.latent_dimension
        )
        z = self.context.to_device(z)

        sample_imgs, _ = self.generator(z)
        for imgs in sample_imgs:
            size = str(imgs.shape[-1])
            imgs = utils.shift_image_range(imgs)
            grid = make_grid(imgs)
            self.logger.writer.add_image(f'generated_sample_images_{size}x{size}', grid, batch_idx)

        self.generator.train()

    def evaluate_full_dataset(self, data_loader: DataLoader) -> Dict[str, Any]:
        self.generator.eval()

        for [z1] in data_loader:
            z1 = self.context.to_device(z1)
            generated_fixed_imgs, _ = self.generator(z1)

            for generated_fixed_img, instability_metric in zip(generated_fixed_imgs, self.instability_metrices):
                instability_metric.add_batch(generated_fixed_img)

        instabilities = []

        for instability_metric in self.instability_metrices:
            instabilities.append(instability_metric())
            instability_metric.step()

        while True:
            z = utils.sample_noise(
                self.context.get_per_slot_batch_size(),
                self.latent_dimension
            )
            z = self.context.to_device(z)

            imgs, _ = self.generator(z)

            if self.inception_score_metric.images is None:
                self.inception_score_metric.images = utils.normalize_image_net(utils.shift_image_range(imgs[-1]))
            else:
                self.inception_score_metric.images = torch.vstack(
                    (self.inception_score_metric.images, utils.normalize_image_net(utils.shift_image_range(imgs[-1])))
                )

            if self.inception_score_metric.images.shape[0] >= self.inception_score_images:
                break

        inception_score, _ = self.inception_score_metric()

        self.generator.train()

        return {
            "inception_score": inception_score,
            **{
                f'{size}x{size}_instability': instability
                for size, instability
                in zip(self.img_sizes, instabilities)
            }
        }

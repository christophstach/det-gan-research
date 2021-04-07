import math
from typing import Any, Dict, Union, List

import torch
from determined.pytorch import PyTorchTrial, PyTorchTrialContext, DataLoader
from determined.tensorboard.metric_writers.pytorch import TorchWriter
from torch import Tensor
from torchvision.utils import make_grid

import datasets as ds
import utils
from metrics import Instability, InceptionScore
from models import MsgDiscriminator, MsgGenerator, ExponentialMovingAverage, BinaryDiscriminator, UnaryDiscriminator
from utils.types import TorchData


class MsgPairGANTrail(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext) -> None:
        super().__init__(context)

        self.context = context
        self.logger = TorchWriter()

        self.dataset = self.context.get_hparam('dataset')
        self.image_size = self.context.get_hparam('image_size')
        self.image_channels = self.context.get_hparam('image_channels')
        self.latent_dimension = self.context.get_hparam('latent_dimension')

        self.alpha_init = self.context.get_hparam('alpha_init')
        self.wait_steps = self.context.get_hparam('wait_steps')
        self.anneal_steps = self.context.get_hparam('anneal_steps')

        self.msg = self.context.get_hparam('msg')
        self.pack = self.context.get_hparam('pack')
        self.ema = self.context.get_hparam('ema')
        self.ema_decay = self.context.get_hparam('ema_decay')
        self.instance_noise_until = self.context.get_hparam('instance_noise_until')
        self.clip_grad_norm = self.context.get_hparam('clip_grad_norm')

        self.g_optimizer = self.context.get_hparam('g_optimizer')
        self.g_lr = self.context.get_hparam('g_lr')
        self.g_b1 = self.context.get_hparam('g_b1')
        self.g_b2 = self.context.get_hparam('g_b2')

        self.d_optimizer = self.context.get_hparam('d_optimizer')
        self.d_lr = self.context.get_hparam('d_lr')
        self.d_b1 = self.context.get_hparam('d_b1')
        self.d_b2 = self.context.get_hparam('d_b2')

        self.g_depth = self.context.get_hparam('g_depth')
        self.g_min_filters = self.context.get_hparam('g_min_filters')
        self.g_max_filters = self.context.get_hparam('g_max_filters')

        self.d_depth = self.context.get_hparam('d_depth')
        self.d_min_filters = self.context.get_hparam('d_min_filters')
        self.d_max_filters = self.context.get_hparam('d_max_filters')

        self.g_activation_fn = self.context.get_hparam('g_activation_fn')
        self.g_normalization = self.context.get_hparam('g_normalization')
        self.g_spectral_normalization = self.context.get_hparam('g_spectral_normalization')

        self.d_activation_fn = self.context.get_hparam('d_activation_fn')
        self.d_normalization = self.context.get_hparam('d_normalization')
        self.d_spectral_normalization = self.context.get_hparam('d_spectral_normalization')

        self.inception_score_images = self.context.get_hparam('inception_score_images')
        self.evaluation_model = self.context.get_hparam('evaluation_model')
        self.num_log_images = 6
        self.log_images_interval = 1000

        self.same_labels = None
        self.different_labels = None

        generator_model = MsgGenerator(
            depth=self.g_depth,
            min_filters=self.g_min_filters,
            max_filters=self.g_max_filters,
            image_size=self.image_size,
            image_channels=self.image_channels,
            latent_dimension=self.latent_dimension,
            activation_fn=self.g_activation_fn,
            normalization=self.g_normalization,
            spectral_normalization=self.g_spectral_normalization,
            msg=self.msg
        )

        discriminator_model = MsgDiscriminator(
            depth=self.d_depth,
            min_filters=self.d_min_filters,
            max_filters=self.d_max_filters,
            image_size=self.image_size,
            image_channels=self.image_channels,
            activation_fn=self.d_activation_fn,
            normalization=self.d_normalization,
            spectral_normalization=self.d_spectral_normalization,
            msg=self.msg,
            unary=True,
            pack=self.pack
        )

        binary_discriminator_model = BinaryDiscriminator()
        discriminator_model = UnaryDiscriminator(discriminator_model)
        generator_model = ExponentialMovingAverage(generator_model, self.ema_decay) if self.ema else generator_model

        self.discriminator = self.context.wrap_model(discriminator_model)
        self.binary_discriminator = self.context.wrap_model(binary_discriminator_model)
        self.generator = self.context.wrap_model(generator_model)

        self.opt_d = self.context.wrap_optimizer(
            utils.create_optimizer(
                self.d_optimizer,
                list(self.discriminator.parameters()) + list(self.binary_discriminator.parameters()),
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

        self.loss = torch.nn.BCEWithLogitsLoss()

        if self.msg:
            self.image_sizes = [
                2 ** (x + 1)
                for x in range(1, int(math.log2(self.image_size)))
            ]
        else:
            self.image_sizes = [self.image_size]

        eval_model, resize_to, num_classes = utils.create_evaluator(self.evaluation_model)
        eval_model = self.context.wrap_model(eval_model)

        self.fixed_z = utils.sample_noise(
            self.num_log_images,
            self.latent_dimension
        )

        self.instability_metrices = [Instability() for _ in self.image_sizes]
        self.inception_score_metric = InceptionScore(
            eval_model,
            resize_to,
            num_classes,
            self.context.get_per_slot_batch_size()
        )

    def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int) -> Union[Tensor, Dict[str, Any]]:
        assert batch[0].shape[0] % 3 == 0

        real_images, _ = batch
        batch_size = real_images.shape[0]
        divided_batch_size = batch_size // 3

        self.same_labels = torch.full((divided_batch_size // self.pack,), 1.0)
        self.different_labels = torch.full((divided_batch_size // self.pack,), 0.0)

        self.same_labels = self.context.to_device(self.same_labels)
        self.different_labels = self.context.to_device(self.different_labels)

        z1 = utils.sample_noise(divided_batch_size, self.latent_dimension)
        z2 = utils.sample_noise(divided_batch_size, self.latent_dimension)
        z3 = utils.sample_noise(divided_batch_size, self.latent_dimension)

        z1 = self.context.to_device(z1)
        z2 = self.context.to_device(z2)
        z3 = self.context.to_device(z3)

        real_images1 = real_images[:divided_batch_size]
        real_images2 = real_images[divided_batch_size:divided_batch_size * 2]
        real_images3 = real_images[divided_batch_size * 2:divided_batch_size * 3]

        if self.msg:
            real_images1 = utils.to_scaled_images(real_images1, self.image_size)
            real_images2 = utils.to_scaled_images(real_images2, self.image_size)
            real_images3 = utils.to_scaled_images(real_images3, self.image_size)
        else:
            real_images1 = [real_images1]
            real_images2 = [real_images2]
            real_images3 = [real_images3]

        d_loss, loss_same_real, loss_same_fake, loss_different = self.optimize_discriminator([
            z1,
            z2,
            z3
        ], [
            real_images1,
            real_images1,
            real_images1
        ])

        g_loss, alpha = self.optimize_generator([
            z1,
            z2,
            z3
        ], [
            real_images1,
            real_images2,
            real_images3
        ], batch_idx)

        if batch_idx % self.log_images_interval == 0:
            self.log_fixed_images(batch_idx)
            self.log_sample_images(batch_idx)

        return {
            'd_loss': d_loss,
            'g_loss': g_loss,
            'alpha': alpha,
            'loss_same_real': loss_same_real,
            'loss_same_fake': loss_same_fake,
            'loss_different': loss_different
        }

    def optimize_discriminator(self, zs: List[Tensor], real_images: List[List[Tensor]]):
        self.discriminator.zero_grad()
        self.binary_discriminator.zero_grad()

        z1, z2, z3 = zs
        real_images1, real_images2, real_images3 = real_images

        fake_images1, w1 = self.generator(z1)
        fake_images2, w2 = self.generator(z2)
        fake_images3, w3 = self.generator(z3)

        real1_scores = self.discriminator(real_images1)
        real2_scores = self.discriminator(real_images2)
        real3_scores = self.discriminator(real_images3)

        fake1_scores = self.discriminator([image.detach() for image in fake_images1])
        fake2_scores = self.discriminator([image.detach() for image in fake_images2])
        fake3_scores = self.discriminator([image.detach() for image in fake_images3])

        # train with same
        same_real_out = self.binary_discriminator(real1_scores + torch.mean(real2_scores))
        same_fake_out = self.binary_discriminator(torch.mean(fake1_scores) + fake2_scores)

        loss_same_real = self.loss(same_real_out, self.same_labels)
        loss_same_fake = self.loss(same_fake_out, self.same_labels)

        # train with different
        different_out = self.binary_discriminator(real3_scores + torch.mean(fake3_scores))
        loss_different = self.loss(different_out, self.different_labels)

        # update discriminator
        d_loss = loss_same_real + loss_same_fake + 2 * loss_different

        self.context.backward(d_loss)
        self.context.step_optimizer(self.opt_d)

        return d_loss, loss_same_real, loss_same_fake, loss_different

    def optimize_generator(self, zs: List[Tensor], real_images: List[List[Tensor]], batch_idx: int):
        self.generator.zero_grad()

        # annealing
        if batch_idx < self.wait_steps:
            alpha = self.alpha_init
        elif batch_idx < self.wait_steps + self.anneal_steps:
            alpha = self.alpha_init * (1. - (batch_idx - self.wait_steps) / self.anneal_steps)
        else:
            alpha = 0.0

        z1, z2, z3 = zs
        real_images1, real_images2, real_images3 = real_images

        fake_images1, w1 = self.generator(z1)
        fake_images2, w2 = self.generator(z2)
        fake_images3, w3 = self.generator(z3)

        real3_scores = self.discriminator(real_images3)

        fake1_scores = self.discriminator(fake_images1)
        fake2_scores = self.discriminator(fake_images2)
        fake3_scores = self.discriminator(fake_images3)

        # train with same fake
        same_fake_out = self.binary_discriminator(torch.mean(fake1_scores) + fake2_scores)

        loss_same_fake_different = self.loss(same_fake_out, self.same_labels)
        loss_same_fake_same = self.loss(same_fake_out, self.different_labels)

        # train with different
        different_out = self.binary_discriminator(real3_scores + torch.mean(fake3_scores))

        loss_different = self.loss(different_out, self.same_labels)

        # update generator
        g_loss = 2 * loss_different + loss_same_fake_different * alpha - loss_same_fake_same * (1 - alpha)

        self.context.backward(g_loss)
        self.context.step_optimizer(self.opt_g)

        if self.ema:
            self.generator.update()

        return g_loss, alpha

    def log_fixed_images(self, batch_idx):
        self.generator.eval()

        fixed_z = self.context.to_device(self.fixed_z)
        fixed_images, _ = self.generator(fixed_z)

        for images in fixed_images:
            size = str(images.shape[-1])
            images = utils.shift_image_range(images)
            grid = make_grid(images)
            self.logger.writer.add_image(f'generated_fixed_images_{size}x{size}', grid, batch_idx)

        self.generator.train()

    def log_sample_images(self, batch_idx):
        self.generator.eval()

        z = utils.sample_noise(
            self.num_log_images,
            self.latent_dimension
        )
        z = self.context.to_device(z)

        sample_images, _ = self.generator(z)
        for images in sample_images:
            size = str(images.shape[-1])
            images = utils.shift_image_range(images)
            grid = make_grid(images)
            self.logger.writer.add_image(f'generated_sample_images_{size}x{size}', grid, batch_idx)

        self.generator.train()

    def build_training_data_loader(self) -> DataLoader:
        train_data = utils.create_dataset(dataset=self.dataset, size=self.image_size, channels=self.image_channels)

        return DataLoader(
            train_data,
            batch_size=self.context.get_per_slot_batch_size() * 3,
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

    def evaluate_full_dataset(self, data_loader: DataLoader) -> Dict[str, Any]:
        self.generator.eval()

        for [z1] in data_loader:
            z1 = self.context.to_device(z1)
            generated_fixed_images, _ = self.generator(z1)

            for generated_fixed_images, instability_metric in zip(generated_fixed_images, self.instability_metrices):
                instability_metric.add_batch(generated_fixed_images)

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

            images, _ = self.generator(z)

            if self.inception_score_metric.images is None:
                self.inception_score_metric.images = utils.normalize_image_net(utils.shift_image_range(images[-1]))
            else:
                self.inception_score_metric.images = torch.vstack(
                    (self.inception_score_metric.images, utils.normalize_image_net(utils.shift_image_range(images[-1])))
                )

            if self.inception_score_metric.images.shape[0] >= self.inception_score_images:
                break

        inception_score, _ = self.inception_score_metric()

        self.generator.train()

        return {
            'inception_score': inception_score,
            **{
                f'{size}x{size}_instability': instability
                for size, instability
                in zip(self.image_sizes, instabilities)
            }
        }

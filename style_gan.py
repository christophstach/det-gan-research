from typing import Union, Dict, Any

import torch
from determined import pytorch
from determined.pytorch import PyTorchTrial, PyTorchTrialContext, DataLoader, TorchData
from determined.tensorboard.metric_writers.pytorch import TorchWriter
from torch import Tensor
from torch.optim import Adam
from torchvision.utils import make_grid

from loss_regularizers.msg_wgan_div_gradient_penalty import MsgWganDivGradientPenalty
from losses.ra_lsgan import RaLSGAN
from metrics import FrechetInceptionDistance
from metrics.inception_score import ClassifierScore
from models.exponential_moving_average import ExponentialMovingAverage
from pl.msg_discriminator import MsgDiscriminator
from pl.msg_generator import MsgGenerator
from utils import shift_image_range, create_dataset, create_evaluator, to_scaled_images
from utils.create_dataset import DatasetSplit


class StyleGanTrial(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext):
        super().__init__(context)

        self.context = context
        self.logger = TorchWriter()
        self.num_log_images = 25

        self.dataset = self.context.get_hparam('dataset')
        self.image_size = self.context.get_hparam('image_size')
        self.image_channels = self.context.get_hparam('image_channels')
        self.latent_dim = self.context.get_hparam('latent_dim')
        self.score_dim = self.context.get_hparam('score_dim')

        self.g_depth = self.context.get_hparam('g_depth')
        self.d_depth = self.context.get_hparam('d_depth')

        self.g_lr = self.context.get_hparam('g_lr')
        self.g_b1 = self.context.get_hparam('g_b1')
        self.g_b2 = self.context.get_hparam('g_b2')

        self.d_lr = self.context.get_hparam('d_lr')
        self.d_b1 = self.context.get_hparam('d_b1')
        self.d_b2 = self.context.get_hparam('d_b2')

        self.generator = MsgGenerator(self.g_depth, self.image_size, self.image_channels, self.latent_dim)
        self.generator = ExponentialMovingAverage(self.generator)
        self.discriminator = MsgDiscriminator(self.d_depth, self.image_size, self.image_channels, self.score_dim)
        self.evaluator, resize_to, num_classes = create_evaluator('vggface2')
        self.evaluator.eval()

        self.g_opt = Adam(self.generator.parameters(), self.g_lr, (self.g_b1, self.g_b2))
        self.d_opt = Adam(self.discriminator.parameters(), self.d_lr, (self.d_b1, self.d_b2))

        self.generator = self.context.wrap_model(self.generator)
        self.discriminator = self.context.wrap_model(self.discriminator)
        self.evaluator = self.context.wrap_model(self.evaluator)

        self.g_opt = self.context.wrap_optimizer(self.g_opt)
        self.d_opt = self.context.wrap_optimizer(self.d_opt)

        self.loss = RaLSGAN()
        self.gradient_penalty = MsgWganDivGradientPenalty(self.discriminator)
        self.fixed_z = torch.randn(self.num_log_images, self.latent_dim, 1, 1)

        self.classifier_score = ClassifierScore(
            classifier=self.evaluator,
            resize_to=resize_to
        )

        self.fid = self.context.experimental.wrap_reducer(
            FrechetInceptionDistance(context),
            name='val_fid',
            for_training=False,
            for_validation=True
        )

    def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int) -> Union[Tensor, Dict[str, Any]]:
        real_images, _ = batch
        batch_size = real_images.shape[0]
        real_images = to_scaled_images(real_images, self.image_size)

        d_logs = self.discriminator_batch(real_images, batch_size)
        g_logs = self.generator_batch(real_images, batch_size)

        self.generator.eval()
        z = torch.randn(batch_size, self.latent_dim, 1, 1)
        z = self.context.to_device(z)
        fake_images = self.generator(z)
        classifier_score = self.classifier_score(fake_images[0])
        self.generator.train()

        return {
            **g_logs,
            **d_logs,
            'classifier_score': classifier_score,
        }

    def generator_batch(self, real_images, batch_size):
        real_images = [r.requires_grad_(False) for r in real_images]

        self.generator.zero_grad()

        z = torch.randn(batch_size, self.latent_dim, 1, 1)
        z = self.context.to_device(z)
        fake_images = self.generator(z)

        real_scores = self.discriminator(real_images)
        fake_scores = self.discriminator(fake_images)

        g_loss = self.loss.generator_loss(real_scores, fake_scores)

        self.context.backward(g_loss)
        self.context.step_optimizer(self.g_opt)
        self.generator.update()

        return {'g_loss': g_loss.item()}

    def discriminator_batch(self, real_images, batch_size):
        real_images = [r.requires_grad_(True) for r in real_images]

        self.discriminator.zero_grad()

        z = torch.randn(batch_size, self.latent_dim, 1, 1, requires_grad=True)
        z = self.context.to_device(z)
        fake_images = self.generator(z)

        real_scores = self.discriminator(real_images)
        fake_scores = self.discriminator(fake_images)

        d_loss = self.loss.discriminator_loss(real_scores, fake_scores)
        gp = self.gradient_penalty(real_images, fake_images, real_scores, fake_scores)

        self.context.backward(d_loss + gp)
        self.context.step_optimizer(self.d_opt)

        return {'d_loss': d_loss.item(), 'gp': gp.item()}

    def evaluate_batch(self, batch: TorchData, batch_idx: int) -> Dict[str, Any]:
        real_images, _ = batch
        batch_size = real_images.shape[0]
        real_images = to_scaled_images(real_images, self.image_size)

        self.generator.eval()
        self.discriminator.eval()

        if batch_idx == 0:
            # log sample images
            z = torch.randn(self.num_log_images, self.latent_dim, 1, 1)
            z = self.context.to_device(z)

            sample_images_list = self.generator(z)
            for sample_images in sample_images_list:
                sample_images = shift_image_range(sample_images)
                sample_grid = make_grid(sample_images, nrow=5)

                self.logger.writer.add_image(
                    f'generated_sample_images_{sample_images.shape[2]}x{sample_images.shape[2]}',
                    sample_grid,
                    self.context.current_train_batch()
                )

            # log fixed images
            self.fixed_z = self.context.to_device(self.fixed_z)
            fixed_images_list = self.generator(self.fixed_z)
            for fixed_images in fixed_images_list:
                fixed_images = shift_image_range(fixed_images)
                fixed_grid = make_grid(fixed_images, nrow=5)

                self.logger.writer.add_image(
                    f'generated_fixed_images_{fixed_images.shape[2]}x{fixed_images.shape[2]}',
                    fixed_grid,
                    self.context.current_train_batch()
                )

        z = torch.randn(batch_size, self.latent_dim, 1, 1)
        z = self.context.to_device(z)
        fake_images = self.generator(z)

        real_scores = self.discriminator(real_images)
        fake_scores = self.discriminator(fake_images)

        val_d_loss = self.loss.discriminator_loss(real_scores, fake_scores)
        val_g_loss = self.loss.generator_loss(real_scores, fake_scores)

        val_classifier_score = self.classifier_score(fake_images[0])
        self.fid.update(real_images[0], fake_images[0])

        self.generator.train()
        self.discriminator.train()

        return {
            'val_d_loss': val_d_loss.item(),
            'val_g_loss': val_g_loss.item(),
            'val_classifier_score': val_classifier_score,
        }

    def evaluation_reducer(self) -> Union[pytorch.Reducer, Dict[str, pytorch.Reducer]]:
        return {
            'val_d_loss': pytorch.Reducer.AVG,
            'val_g_loss': pytorch.Reducer.AVG,
            'val_classifier_score': pytorch.Reducer.AVG,
        }

    def build_training_data_loader(self) -> DataLoader:
        train_data = create_dataset(self.dataset, self.image_size, self.image_channels, DatasetSplit.FULL)

        return DataLoader(
            train_data,
            batch_size=self.context.get_per_slot_batch_size(),
            shuffle=True,
            drop_last=True
        )

    def build_validation_data_loader(self) -> DataLoader:
        validation_data = create_dataset(self.dataset, self.image_size, self.image_channels, DatasetSplit.VALIDATION)

        return DataLoader(
            validation_data,
            batch_size=self.context.get_per_slot_batch_size(),
            shuffle=False,
            drop_last=True
        )

from collections import OrderedDict
from typing import List

import torch
from determined.tensorboard.metric_writers.pytorch import TorchWriter
from pytorch_lightning import LightningModule
from torch import nn, Tensor
from torchvision.utils import make_grid
from typing_extensions import TypedDict

from loss_regularizers.msg_wgan_div_gradient_penalty import MsgWganDivGradientPenalty
from losses import RaLSGAN
from losses.base import Loss
from models import ExponentialMovingAverage
from pl.msg_discriminator import MsgDiscriminator
from pl.msg_generator import MsgGenerator
from utils import create_optimizer, to_scaled_images, shift_image_range


class MsgGanConfig(TypedDict):
    g_lr: float
    g_b1: float
    g_b2: float

    d_lr: float
    d_b1: float
    d_b2: float

    g_depth: int
    d_depth: int

    latent_dim: int
    score_dim: int
    image_channels: int
    image_size: int


class LightningMsgGan(LightningModule):
    cfg: MsgGanConfig
    generator: nn.Module
    discriminator: nn.Module
    loss: Loss
    gradient_penalty: MsgWganDivGradientPenalty

    def __init__(self, cfg: MsgGanConfig, torch_writer: TorchWriter):
        super().__init__()

        self.automatic_optimization = False
        self.cfg = cfg
        self.torch_writer = torch_writer
        self.num_log_images = 25

        self.generator = MsgGenerator(
            self.cfg['g_depth'],
            self.cfg['image_size'],
            self.cfg['image_channels'],
            self.cfg['latent_dim']
        )
        # self.generator = ExponentialMovingAverage(self.generator)
        self.discriminator = MsgDiscriminator(
            self.cfg['d_depth'],
            self.cfg['image_size'],
            self.cfg['image_channels'],
            self.cfg['score_dim']
        )

        self.loss = RaLSGAN()
        self.gradient_penalty = MsgWganDivGradientPenalty(self.discriminator)
        self.fixed_z = torch.randn(self.num_log_images, self.cfg['latent_dim'], 1, 1, device=self.device)

    def forward(self, x):
        return self.generator(x)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_images, _ = batch
        batch_size = real_images.shape[0]
        real_images = to_scaled_images(real_images, self.cfg['image_size'])
        # g_opt, d_opt = self.optimizers()

        if optimizer_idx == 0:
            # self.generator.update()
            result = self.generator_step(real_images, batch_size)
            # self.manual_backward(result['loss'], g_opt)
            # g_opt.step()
            # g_opt.zero_grad()
            # self.generator.update()
        else:
            result = self.discriminator_step(real_images, batch_size)
            # self.manual_backward(result['loss'], d_opt)
            # d_opt.step()
            # d_opt.zero_grad()

        return result

    def generator_step(self, real_images: List[Tensor], batch_size: int):
        self.generator.zero_grad()
        z = torch.randn(batch_size, self.cfg['latent_dim'], 1, 1, device=self.device)
        fake_images = self.generator(z)

        real_scores = self.discriminator(real_images)
        fake_scores = self.discriminator(fake_images)

        g_loss = self.loss.generator_loss(real_scores, fake_scores)

        return OrderedDict({
            'loss': g_loss,
            'log': {
                'g_loss': g_loss.item()
            }
        })

    def discriminator_step(self, real_images: List[Tensor], batch_size: int):
        real_images = [r.requires_grad_(True) for r in real_images]

        self.discriminator.zero_grad()
        z = torch.randn(batch_size, self.cfg['latent_dim'], 1, 1, device=self.device, requires_grad=True)
        fake_images = self.generator(z)

        real_scores = self.discriminator(real_images)
        fake_scores = self.discriminator(fake_images)

        d_loss = self.loss.discriminator_loss(real_scores, fake_scores)
        gp = self.gradient_penalty(real_images, fake_images, real_scores, fake_scores)

        return OrderedDict({
            'loss': d_loss + gp,
            'log': {
                'd_loss': d_loss.item(),
                'gp': gp.item()
            }
        })

    def validation_step(self, batch, batch_idx):
        real_images, _ = batch
        batch_size = real_images.shape[0]

        self.generator.eval()
        self.discriminator.eval()

        if batch_idx == 0:
            # log sample images
            z = torch.randn(self.num_log_images, self.cfg['latent_dim'], 1, 1, device=self.device)

            sample_images_list = self.generator(z)
            for sample_images in sample_images_list:
                sample_images = shift_image_range(sample_images)
                sample_grid = make_grid(sample_images, nrow=5)

                self.torch_writer.writer.add_image(
                    f'generated_sample_images_{sample_images.shape[2]}x{sample_images.shape[2]}',
                    sample_grid,
                    self.global_step
                )

            # log fixed images
            fixed_images_list = self.generator(self.fixed_z.to(device=self.device))
            for fixed_images in fixed_images_list:
                fixed_images = shift_image_range(fixed_images)
                fixed_grid = make_grid(fixed_images, nrow=5)

                self.torch_writer.writer.add_image(
                    f'generated_fixed_images_{fixed_images.shape[2]}x{fixed_images.shape[2]}',
                    fixed_grid,
                    self.global_step
                )

        self.generator.train()
        self.discriminator.train()

        return {
            'val_classifier_score': 0
        }

    def configure_optimizers(self):
        return (
            {
                'optimizer': create_optimizer(
                    'adam',
                    self.generator.parameters(),
                    self.cfg['g_lr'],
                    (self.cfg['g_b1'], self.cfg['g_b2'])
                ),
                'frequency': 1
            },
            {
                'optimizer': create_optimizer(
                    'adam',
                    self.discriminator.parameters(),
                    self.cfg['d_lr'],
                    (self.cfg['d_b1'], self.cfg['d_b2'])
                ),
                'frequency': 1
            }
        )

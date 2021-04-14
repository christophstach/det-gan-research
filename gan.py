from typing import Union, Dict, Any

from determined import pytorch
from determined.pytorch import PyTorchTrial, PyTorchTrialContext, DataLoader, TorchData
from determined.tensorboard.metric_writers.pytorch import TorchWriter
from torch import Tensor
from torch.optim import Adam, SGD, RMSprop
from torchvision.utils import make_grid
from optim import OAdam

from loss_regularizers.simple_wgan_div_gradient_penalty import GradientPenalty
from losses.ra_lsgan import RaLSGAN
from metrics.inception_score import ClassifierScore
from models.exponential_moving_average import ExponentialMovingAverage
from models.optimal_discriminator import OptimalDiscriminator
from models.skip_generator import SkipGenerator
from utils import shift_image_range, create_dataset, sample_noise, create_evaluator


class GANTrail(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext) -> None:
        super().__init__(context)

        self.context = context
        self.logger = TorchWriter()
        self.num_log_images = 25

        self.dataset = self.context.get_hparam('dataset')
        self.image_size = self.context.get_hparam('image_size')
        self.image_channels = self.context.get_hparam('image_channels')
        self.latent_dimension = self.context.get_hparam('latent_dimension')

        self.g_depth = self.context.get_hparam('g_depth')
        self.d_depth = self.context.get_hparam('d_depth')

        self.g_lr = self.context.get_hparam('g_lr')
        self.g_b1 = self.context.get_hparam('g_b1')
        self.g_b2 = self.context.get_hparam('g_b2')

        self.d_lr = self.context.get_hparam('d_lr')
        self.d_b1 = self.context.get_hparam('d_b1')
        self.d_b2 = self.context.get_hparam('d_b2')

        self.generator = SkipGenerator(self.g_depth, self.image_channels, self.latent_dimension)
        self.generator = ExponentialMovingAverage(self.generator)
        self.discriminator = OptimalDiscriminator(self.d_depth, self.image_channels)
        self.evaluator, resize_to, num_classes = create_evaluator('vggface2')
        self.evaluator.eval()

        # self.g_opt = Adam(self.generator.parameters(), self.g_lr, (self.g_b1, self.g_b2))
        # self.d_opt = Adam(self.discriminator.parameters(), self.d_lr, (self.d_b1, self.d_b2))
        # self.g_opt = RMSprop(self.generator.parameters(), self.g_lr)
        # self.d_opt = RMSprop(self.discriminator.parameters(), self.d_lr)
        self.g_opt = OAdam(self.generator.parameters(), self.g_lr, (self.g_b1, self.g_b2))
        self.d_opt = OAdam(self.discriminator.parameters(), self.d_lr, (self.d_b1, self.d_b2))

        self.generator = self.context.wrap_model(self.generator)
        self.discriminator = self.context.wrap_model(self.discriminator)
        self.evaluator = self.context.wrap_model(self.evaluator)

        self.g_opt = self.context.wrap_optimizer(self.g_opt)
        self.d_opt = self.context.wrap_optimizer(self.d_opt)

        self.loss = RaLSGAN()
        self.gradient_penalty = GradientPenalty(self.discriminator)

        self.fixed_z = sample_noise(self.num_log_images, self.latent_dimension)

        self.classifier_score = ClassifierScore(
            classifier=self.evaluator,
            resize_to=resize_to
        )

    def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int) -> Union[Tensor, Dict[str, Any]]:
        real_images, _ = batch
        batch_size = real_images.shape[0]
        real_images.requires_grad_(True)

        # optimize discriminator
        self.discriminator.zero_grad()
        z = sample_noise(batch_size, self.latent_dimension)
        z = self.context.to_device(z)

        # with no_grad():
        fake_images = self.generator(z)

        real_scores = self.discriminator(real_images)
        fake_scores = self.discriminator(fake_images)

        d_loss = self.loss.discriminator_loss(real_scores, fake_scores)
        gp = self.gradient_penalty(real_images, fake_images, real_scores, fake_scores)

        self.context.backward(d_loss + gp)
        self.context.step_optimizer(self.d_opt)

        # optimize generator
        self.generator.zero_grad()
        z = sample_noise(batch_size, self.latent_dimension)
        z = self.context.to_device(z)

        fake_images = self.generator(z)

        real_scores = self.discriminator(real_images)
        fake_scores = self.discriminator(fake_images)

        g_loss = self.loss.generator_loss(real_scores, fake_scores)

        self.context.backward(g_loss)
        self.context.step_optimizer(self.g_opt)
        self.generator.update()

        self.generator.eval()
        classifier_score = self.classifier_score(self.generator(z))
        self.generator.train()

        return {
            'd_loss': d_loss,
            'g_loss': g_loss,
            'classifier_score': classifier_score,
            'gp': gp
        }

    def evaluate_batch(self, batch: TorchData, batch_idx: int) -> Dict[str, Any]:
        real_images, _ = batch
        batch_size = real_images.shape[0]

        self.generator.eval()
        self.discriminator.eval()

        if batch_idx == 0:
            # log sample images
            z = sample_noise(self.num_log_images, self.latent_dimension)
            z = self.context.to_device(z)

            sample_images = self.generator(z)
            sample_images = shift_image_range(sample_images)
            sample_grid = make_grid(sample_images, nrow=5)

            self.logger.writer.add_image(f'generated_sample_images', sample_grid, self.context.current_train_batch())

            # log fixed images
            z = self.context.to_device(self.fixed_z)

            fixed_images = self.generator(z)
            fixed_images = shift_image_range(fixed_images)
            fixed_grid = make_grid(fixed_images, nrow=5)

            self.logger.writer.add_image(f'generated_fixed_images', fixed_grid, self.context.current_train_batch())

            self.generator.train()

        z = sample_noise(batch_size, self.latent_dimension)
        z = self.context.to_device(z)

        fake_images = self.generator(z)

        real_scores = self.discriminator(real_images)
        fake_scores = self.discriminator(fake_images)

        val_d_loss = self.loss.discriminator_loss(real_scores, fake_scores)
        val_g_loss = self.loss.generator_loss(real_scores, fake_scores)

        val_classifier_score = self.classifier_score(self.generator(z))

        self.generator.train()
        self.discriminator.train()

        return {
            'val_d_loss': val_d_loss,
            'val_g_loss': val_g_loss,
            'val_classifier_score': val_classifier_score,
            # 'val_fid': ('random', 'activation')
        }

    def evaluation_reducer(self) -> Union[pytorch.Reducer, Dict[str, pytorch.Reducer]]:
        return {
            'val_d_loss': pytorch.Reducer.AVG,
            'val_g_loss': pytorch.Reducer.AVG,
            'val_classifier_score': pytorch.Reducer.AVG,
            # 'val_fid': pytorch.Reducer.AVG
        }

    def build_training_data_loader(self) -> DataLoader:
        train_data = create_dataset(self.dataset, self.image_size, self.image_channels, True)

        return DataLoader(
            train_data,
            batch_size=self.context.get_per_slot_batch_size(),
            shuffle=True,
            drop_last=True
        )

    def build_validation_data_loader(self) -> DataLoader:
        validation_data = create_dataset(self.dataset, self.image_size, self.image_channels)

        return DataLoader(
            validation_data,
            batch_size=self.context.get_per_slot_batch_size(),
            shuffle=False,
            drop_last=True
        )

from determined.pytorch import PyTorchTrialContext, DataLoader
from determined.pytorch.lightning import LightningAdapter
from determined.tensorboard.metric_writers.pytorch import TorchWriter

from pl.lightning_celeba_hq_data_module import LightningCelebaHqDataModule
from pl.lightning_msg_gan import LightningMsgGan


class PlMsgGanTrial(LightningAdapter):
    def __init__(self, context: PyTorchTrialContext) -> None:
        cfg = {
            'g_lr': context.get_hparam('g_lr'),
            'g_b1': context.get_hparam('g_b1'),
            'g_b2': context.get_hparam('g_b2'),

            'd_lr': context.get_hparam('d_lr'),
            'd_b1': context.get_hparam('d_b1'),
            'd_b2': context.get_hparam('d_b2'),

            'g_depth': context.get_hparam('g_depth'),
            'd_depth': context.get_hparam('d_depth'),

            'latent_dim': context.get_hparam('latent_dim'),
            'score_dim': context.get_hparam('score_dim'),
            'image_channels': context.get_hparam('image_channels'),
            'image_size': context.get_hparam('image_size')
        }

        lm = LightningMsgGan(cfg, TorchWriter())

        # instantiate your LightningDataModule and make it distributed training ready.
        data_dir = f'/datasets'
        self.dm = LightningCelebaHqDataModule(
            cfg['image_size'],
            cfg['image_channels'],
            data_dir,
            context.get_per_slot_batch_size()
        )

        # initialize LightningAdapter.
        super().__init__(context, lightning_module=lm)
        self.dm.prepare_data()

    def build_training_data_loader(self) -> DataLoader:
        self.dm.setup()
        dl = self.dm.train_dataloader()

        return DataLoader(
            dl.dataset, batch_size=dl.batch_size, num_workers=dl.num_workers
        )

    def build_validation_data_loader(self) -> DataLoader:
        self.dm.setup()
        dl = self.dm.val_dataloader()

        return DataLoader(
            dl.dataset, batch_size=dl.batch_size, num_workers=dl.num_workers
        )

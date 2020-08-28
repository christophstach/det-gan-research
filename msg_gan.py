from typing import Union, Any, Dict, Sequence

import torch
import torch.optim as optim
from determined.pytorch import PyTorchTrial, PyTorchTrialContext, DataLoader

from models import MsgDiscriminator, MsgGenerator


class MsgGanTrail(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext) -> None:
        super().__init__(context)

        self.context = context

        lr = self.context.get_hparam("lr")
        b1 = self.context.get_hparam("b1")
        b2 = self.context.get_hparam("b2")

        filter_multiplier = self.context.get_hparam("filter_multiplier")
        min_filters = self.context.get_hparam("min_filters")
        max_filters = self.context.get_hparam("max_filters")
        image_size = self.context.get_hparam("image_size")
        image_channels = self.context.get_hparam("image_channels")
        latent_dimension = self.context.get_hparam("latent_dimension")
        spectral_normalization = self.context.get_hparam("spectral_normalization")

        self.generator = self.context.wrap_model(MsgGenerator(
            filter_multiplier=filter_multiplier,
            min_filters=min_filters,
            max_filters=max_filters,
            image_size=image_size,
            image_channels=image_channels,
            latent_dimension=latent_dimension,
            spectral_normalization=spectral_normalization
        ))
        self.discriminator = self.context.wrap_model(MsgDiscriminator(
            filter_multiplier=filter_multiplier,
            min_filters=min_filters,
            max_filters=max_filters,
            image_size=image_size,
            image_channels=image_channels,
            spectral_normalization=spectral_normalization
        ))

        self.opt_g = self.context.wrap_optimizer(optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2)))
        self.opt_d = self.context.wrap_optimizer(optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2)))

    def train_batch(
            self,
            batch: Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor],
            epoch_idx: int,
            batch_idx: int,
            **kwargs
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        pass

    def build_training_data_loader(self) -> DataLoader:
        pass

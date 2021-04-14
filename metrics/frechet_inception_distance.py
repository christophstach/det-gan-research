from typing import Any, List, Union

import numpy as np
import torch
from determined.pytorch import MetricReducer, PyTorchTrialContext
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from torch import Tensor
from torch.nn.functional import adaptive_avg_pool2d


class FrechetInceptionDistance(MetricReducer):

    def __init__(self, context: PyTorchTrialContext) -> None:
        super().__init__()

        dims = 2048
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.model = InceptionV3([block_idx])
        self.model = context.wrap_model(self.model)

        self.real_activations = None
        self.fake_activations = None

        self.reset()

    def reset(self) -> None:
        self.real_activations = []
        self.fake_activations = []

    def get_activations(self, images):
        with torch.no_grad():
            predictions = self.model(images)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if predictions.size(2) != 1 or predictions.size(3) != 1:
            predictions = adaptive_avg_pool2d(predictions, output_size=(1, 1))

        predictions = predictions.view(predictions.shape[0], -1)

        return predictions

    def update(self, real_images: Tensor, fake_images: Tensor):
        self.real_activations = self.get_activations(real_images)
        self.fake_activations = self.get_activations(fake_images)

    def per_slot_reduce(self) -> Any:
        return self.real_activations, self.fake_activations

    def cross_slot_reduce(self, per_slot_metrics: List) -> Any:
        real_activations: Union[Tensor, None] = None
        fake_activations: Union[Tensor, None] = None

        for real, fake in per_slot_metrics:
            real_activations = real if real_activations is None else torch.vstack([real_activations, real])
            fake_activations = fake if fake_activations is None else torch.vstack([fake_activations, fake])

        real_activations = real_activations.cpu().numpy()
        fake_activations = fake_activations.cpu().numpy()

        real_mu = np.mean(real_activations, axis=0)
        real_sigma = np.cov(real_activations, rowvar=False)

        fake_mu = np.mean(fake_activations, axis=0)
        fake_sigma = np.cov(fake_activations, rowvar=False)

        return calculate_frechet_distance(real_mu, real_sigma, fake_mu, fake_sigma)

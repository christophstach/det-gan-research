from typing import List, Any

import numpy as np
import torch
import torch.nn.functional as F
from determined.pytorch import DataLoader
from determined.pytorch import MetricReducer
from scipy.stats import entropy
from torch import nn

from .base import Metric


class ClassifierScoreReducer(MetricReducer):

    def reset(self) -> None:
        pass

    def per_slot_reduce(self) -> Any:
        pass

    def cross_slot_reduce(self, per_slot_metrics: List) -> Any:
        pass


# https://github.com/torchgan/torchgan/blob/master/torchgan/metrics/classifierscore.py
class ClassifierScore:
    def __init__(self, classifier: nn.Module, resize_to: int) -> None:
        super().__init__()

        self.classifier = classifier
        self.resize_to = resize_to

    def resize(self, x):
        return F.interpolate(
            input=x,
            size=(self.resize_to, self.resize_to),
            mode='bilinear',
            align_corners=False
        )

    def __call__(self, x):
        x = self.resize(x) if x.shape[2] != self.resize_to or x.shape[3] != self.resize_to else x
        x = self.classifier(x)

        p = F.softmax(x, dim=1)
        q = torch.mean(p, dim=0)

        kl = torch.sum(p * (F.log_softmax(x, dim=1) - torch.log(q)), dim=1)
        return torch.exp(torch.mean(kl)).data


# https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py
class InceptionScore(Metric):
    def __init__(self,
                 model: torch.nn.Module,
                 resize_to: int,
                 num_classes: int,
                 batch_size: int = 32,
                 resize: bool = True,
                 splits: int = 1
                 ) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.splits = splits
        self.resize = resize
        self.model = model
        self.resize_to = resize_to
        self.num_classes = num_classes

        self.images = None

    def predict(self, x):
        x = F.interpolate(
            x,
            size=(self.resize_to, self.resize_to),
            mode="bilinear",
            align_corners=False
        ) if self.resize else x

        x = self.model(x)

        return F.softmax(x, dim=-1).detach().cpu().numpy()

    def __call__(self):
        if self.images is None:
            raise ValueError("Must set images first to be able to calculate IC!")

        num_images = len(self.images)

        assert self.batch_size > 0
        assert num_images >= self.batch_size

        dataloader = DataLoader(self.images, batch_size=self.batch_size)

        predictions = np.zeros((num_images, self.num_classes))

        for i, batch in enumerate(dataloader, 0):
            batch_size_i = batch.size()[0]

            predictions[i * self.batch_size:i * self.batch_size + batch_size_i] = self.predict(batch)

        # Now compute the mean kl-div
        split_scores = []

        for k in range(self.splits):
            part = predictions[k * (num_images // self.splits): (k + 1) * (num_images // self.splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))

            split_scores.append(np.exp(np.mean(scores)))

        self.images = None

        return np.mean(split_scores), np.std(split_scores)

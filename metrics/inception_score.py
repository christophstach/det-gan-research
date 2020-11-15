import numpy as np
import torch
import torch.nn.functional as F
from determined.pytorch import DataLoader
from scipy.stats import entropy

from .base import Metric


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
        assert num_images > self.batch_size

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

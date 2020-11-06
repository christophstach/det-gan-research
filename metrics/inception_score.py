import numpy as np
import torch.nn.functional as F
from determined.pytorch import DataLoader
from scipy.stats import entropy

from .base import Metric


# https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py
class InceptionScore(Metric):

    def __init__(self, inception_model, batch_size=32, resize=True, splits=1) -> None:
        super().__init__()

        self.inception_model = inception_model
        self.batch_size = batch_size
        self.resize = resize
        self.splits = splits

        self.images = None

    def predict(self, x):
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False) if self.resize else x
        x = self.inception_model(x)

        return F.softmax(x, dim=-1).detach().cpu().numpy()

    def __call__(self):
        if self.images is None:
            raise ValueError("Must set images first to be able to calculate IC!")

        N = len(self.images)

        assert self.batch_size > 0
        assert N > self.batch_size

        dataloader = DataLoader(self.images, batch_size=self.batch_size)

        predictions = np.zeros((N, 1000))

        for i, batch in enumerate(dataloader, 0):
            batch_size_i = batch.size()[0]

            predictions[i * self.batch_size:i * self.batch_size + batch_size_i] = self.predict(batch)

        # Now compute the mean kl-div
        split_scores = []

        for k in range(self.splits):
            part = predictions[k * (N // self.splits): (k + 1) * (N // self.splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))

            split_scores.append(np.exp(np.mean(scores)))

        self.images = None

        return np.mean(split_scores), np.std(split_scores)

import torch
import torch.nn.functional as F

from .base import Metric


class Instability(Metric):
    def __init__(self) -> None:
        super().__init__()

        self.last_data = []
        self.current_data = []

    def add_batch(self, batch):
        self.current_data.append(batch)

    def step(self) -> None:
        self.last_data = self.current_data
        self.current_data = []

    def __call__(self):
        if len(self.last_data) == 0:
            return 0
        elif len(self.last_data) == len(self.current_data):
            instabilities = [
                F.mse_loss(last_batch, current_batch)
                for last_batch, current_batch
                in zip(self.last_data, self.current_data)
            ]

            return torch.mean(torch.tensor(instabilities)).item()
        else:
            raise ValueError("Size of last_data and current_data is not equal")

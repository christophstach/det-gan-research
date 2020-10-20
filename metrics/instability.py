from .base import Metric
import torch
import torch.nn.functional as F
import math


class Instability(Metric):
    def __init__(self) -> None:
        super().__init__()

        self.__last_data = []
        self.__current_data = []

    def add_batch(self, batch):
        self.__current_data.append(batch)

    def step(self):
        self.__last_data = self.__current_data
        self.__current_data = []

    def __call__(self):
        if len(self.__last_data) == 0:
            return 0
        elif len(self.__last_data) == len(self.__current_data):
            stabilities = [
                F.l1_loss(last_batch, current_batch)
                for last_batch, current_batch
                in zip(self.__last_data, self.__current_data)
            ]

            return torch.mean(torch.tensor(stabilities)).item()
        else:
            raise ValueError("Size of last_data and current_data is not equal")

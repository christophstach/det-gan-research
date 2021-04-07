from .base import Metric
from typing import Any, List

from determined.pytorch import MetricReducer


class FrechetInceptionDistance(MetricReducer):

    def __init__(self) -> None:
        super().__init__()

        self.real_activations = None
        self.fake_activations = None

        self.reset()

    def reset(self) -> None:
        self.real_activations = []
        self.fake_activations = []

    def per_slot_reduce(self) -> Any:
        pass

    def cross_slot_reduce(self, per_slot_metrics: List) -> Any:
        pass


class FrechetInceptionDistance(Metric):

    def __call__(self):
        pass

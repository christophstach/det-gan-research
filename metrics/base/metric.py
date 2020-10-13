import abc


class Metric(abc.ABC):
    @abc.abstractmethod
    def __call__(self):
        return None

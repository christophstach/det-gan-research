from enum import Enum

from torch.utils.data import random_split

import datasets as ds


class DatasetSplit(Enum):
    FULL = 'full'
    TRAIN = 'train'
    VALIDATION = 'validation'
    TEST = 'test'


def create_dataset(dataset: str, size: int, channels: int = None, split: DatasetSplit = DatasetSplit.FULL):
    def mnist():
        if channels:
            ds_full = ds.mnist(size, channels, train=(split != DatasetSplit.TEST))
        else:
            ds_full = ds.mnist(size, train=(split != DatasetSplit.TEST))

        ds_train, ds_validation = random_split(ds_full, [50000, 10000])

        if split == DatasetSplit.TRAIN:
            return ds_train
        elif split == DatasetSplit.VALIDATION:
            return ds_validation
        else:
            return ds_full

    def ffhq():
        if channels:
            ds_full = ds.ffhq(size, channels)
        else:
            ds_full = ds.ffhq(size)

        ds_train, ds_validation = random_split(ds_full, [60000, 10000])

        if split == DatasetSplit.TRAIN:
            return ds_train
        elif split == DatasetSplit.VALIDATION:
            return ds_validation
        elif split == DatasetSplit.TEST:
            raise ValueError('No test set available')
        else:
            return ds_full

    def celeba_hq():
        if channels:
            ds_full = ds.celeba_hq(size, channels)
        else:
            ds_full = ds.celeba_hq(size)

        ds_train, ds_validation = random_split(ds_full, [20000, 10000])

        if split == DatasetSplit.TRAIN:
            return ds_train
        elif split == DatasetSplit.VALIDATION:
            return ds_validation
        elif split == DatasetSplit.TEST:
            raise ValueError('No test set available')
        else:
            return ds_full

    def anime_face():
        if channels:
            ds_full = ds.anime_face(size, channels)
        else:
            ds_full = ds.anime_face(size)

        ds_train, ds_validation = random_split(ds_full, [53569, 10000])

        if split == DatasetSplit.TRAIN:
            return ds_train
        elif split == DatasetSplit.VALIDATION:
            return ds_validation
        elif split == DatasetSplit.TEST:
            raise ValueError('No test set available')
        else:
            return ds_full

    dataset_dict = {
        "mnist": mnist,
        "ffhq": ffhq,
        "celeba-hq": celeba_hq,
        "anime-face": anime_face
    }

    return dataset_dict[dataset]()

import datasets as ds


def create_dataset(dataset: str, size: int, channels: int = None):
    dataset_dict = {
        "mnist": lambda: ds.mnist(True, size, channels) if channels else ds.mnist(True, size),
        "ffhq": lambda: ds.ffhq(size, channels) if channels else ds.ffhq(size),
        "celeba-hq": lambda: ds.celeba_hq(size, channels) if channels else ds.celeba_hq(size),
        "anime-face": lambda: ds.anime_face(size, channels) if channels else ds.anime_face(size),
    }

    return dataset_dict[dataset]()

import datasets as ds


def create_dataset(dataset: str, size: int, channels: int = None, validation: bool = False):
    def mnist():
        if channels:
            return ds.mnist(size, channels, validation=validation)
        else:
            return ds.mnist(size, validation=validation)

    def ffhq():
        if channels:
            return ds.ffhq(size, channels, validation=False)
        else:
            return ds.ffhq(size, validation=False)

    def celeba_hq():
        if channels:
            return ds.celeba_hq(size, channels, validation=validation)
        else:
            return ds.celeba_hq(size, validation=validation)

    def anime_face():
        if channels:
            return ds.anime_face(size, channels, validation=validation)
        else:
            return ds.anime_face(size, validation=validation)

    dataset_dict = {
        "mnist": mnist,
        "ffhq": ffhq,
        "celeba-hq": celeba_hq,
        "anime-face": anime_face
    }

    return dataset_dict[dataset]()

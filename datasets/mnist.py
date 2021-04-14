from torchvision import datasets, transforms


def mnist(size=28, channels=1, root="/datasets", train=True):
    assert channels == 1 or channels == 3

    transform_ops = [
        None if size == 28 else transforms.Resize(size),
        transforms.ToTensor(),
        None if channels == 1 else transforms.Lambda(lambda img: img.repeat([3, 1, 1])),
        transforms.Normalize((0.5,), (0.5,)) if channels == 1
        else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    transform = transforms.Compose([op for op in transform_ops if op is not None])

    return datasets.MNIST(
        root,
        train=train,
        download=False,
        transform=transform
    )

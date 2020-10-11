from torchvision import datasets, transforms


def mnist(train=True, size=28):
    transform_ops = [
        None if size == 28 else transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]

    transform = transforms.Compose([op for op in transform_ops if op is not None])

    return datasets.MNIST(
        "/datasets",
        train=train,
        download=False,
        transform=transform
    )

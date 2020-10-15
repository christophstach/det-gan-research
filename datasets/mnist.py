from torchvision import datasets, transforms
import torch


def mnist(train=True, size=28, channels=1, root="/datasets"):
    def repeat_channel_dimension(img: torch.Tensor) -> torch.Tensor:
        return img.repeat([3, 1, 1])

    transform_ops = [
        None if size == 28 else transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        None if channels == 1 else transforms.Lambda(repeat_channel_dimension)
    ]

    transform = transforms.Compose([op for op in transform_ops if op is not None])

    return datasets.MNIST(
        root,
        train=train,
        download=False,
        transform=transform
    )

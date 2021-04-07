from torchvision import transforms

from .flat_image_folder import FlatImageFolder


def anime_face(size=128, channels=3, root="/datasets", validation=False):
    assert channels == 1 or channels == 3

    root += "/anime-face/cropped"

    transform_ops = [
        transforms.Resize(size),
        transforms.RandomCrop(size),
        None if channels == 3 else transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) if channels == 1
        else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    transform = transforms.Compose([op for op in transform_ops if op is not None])

    return FlatImageFolder(
        root,
        transform=transform
    )

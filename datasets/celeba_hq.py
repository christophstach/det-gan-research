from torchvision import transforms

from .flat_image_folder import FlatImageFolder


def celeba_hq(size=128, channels=1, root="/datasets"):
    assert channels == 1 or channels == 3

    available_sizes = [128, 256, 512, 1024]
    needs_resize = False

    if size not in available_sizes:
        needs_resize = True
        nearest_size = min(filter(lambda s: s > size, available_sizes))
        root += "/celebAHQ/data" + str(nearest_size) + "x" + str(nearest_size)
    else:
        root += "/celebAHQ/data" + str(size) + "x" + str(size)

    transform_ops = [
        None if needs_resize else transforms.Resize(size),
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

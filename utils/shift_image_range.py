import torch


def shift_image_range(images: torch.Tensor, range_in=(-1, 1), range_out=(0, 1)):
    images = images.clone()
    images.detach_()

    scale = (range_out[1] - range_out[0]) / (range_in[1] - range_in[0])
    bias = range_out[0] - range_in[0] * scale
    images = images * scale + bias
    images.clamp_(min=range_out[0], max=range_out[1])

    return images

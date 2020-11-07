import torch
from torchvision import transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def normalize_image_net(tensor: torch.Tensor):
    return normalize(tensor)

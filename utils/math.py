import torch
import math


# https://discuss.pytorch.org/t/how-to-compute-jacobian-matrix-in-pytorch/14968/14

def gradient(y, x, grad_outputs=None):
    """Compute dy/dx @ grad_outputs"""
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)

    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def jacobian(y, x):
    """Compute dy/dx = dy/dx @ grad_outputs;
    for grad_outputs in [1, 0, ..., 0], [0, 1, 0, ..., 0], ...., [0, ..., 0, 1]"""
    jac = torch.zeros(y.shape[0], x.shape[0])

    for i in range(y.shape[0]):
        grad_outputs = torch.zeros_like(y)
        grad_outputs[i] = 1
        jac[i] = gradient(y, x, grad_outputs=grad_outputs)

    return jac


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i + 1]
    return div


def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


# https://github.com/rosinality/stylegan2-pytorch/blob/master/train.py

def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )

    grad = torch.autograd.grad(
        outputs=(fake_img * noise).sum(),
        inputs=latents,
        create_graph=True
    )[0]

    path_lengths = (grad ** 2).sum(dim=2).mean(dim=1).sqrt()
    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)
    path_penalty = ((path_lengths - path_mean) ** 2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def calc_pl_lengths(styles, images):
    num_pixels = images.shape[2] * images.shape[3]
    pl_noise = torch.randn(images.shape) / math.sqrt(num_pixels)
    outputs = (images * pl_noise).sum()

    pl_grads = torch.autograd.grad(
        outputs=outputs,
        inputs=styles,
        grad_outputs=torch.ones(outputs.shape, ),
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    return (pl_grads ** 2).sum(dim=2).mean(dim=1).sqrt()

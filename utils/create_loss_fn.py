from losses import WGAN, RaLSGAN, RaSGAN, RaHinge
from losses.realness import Realness


def create_loss_fn(loss_fn):
    loss_fn_dict = {
        'WGAN': lambda: WGAN(),
        'RaHinge': lambda: RaHinge(),
        'RaLSGAN': lambda: RaLSGAN(),
        'RaSGAN': lambda: RaSGAN(),
        'RealnessRaSGAN': lambda: Realness()
    }

    return loss_fn_dict[loss_fn]()

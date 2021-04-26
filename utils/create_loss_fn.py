from losses import WGAN, RaLSGAN, RaSGAN, RaHinge
from losses.realness import Realness


def create_loss_fn(loss_fn, score_dim):
    loss_fn_dict = {
        'WGAN': lambda: WGAN(),
        'RaHinge': lambda: RaHinge(),
        'RaLSGAN': lambda: RaLSGAN(),
        'RaSGAN': lambda: RaSGAN(),
        'Realness': lambda: Realness(score_dim)
    }

    return loss_fn_dict[loss_fn]()

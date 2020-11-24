from torch.optim import Adam
from optim import OAdam


def create_optimizer(optimizer, parameters, lr, betas):
    optimizer_dict = {
        "Adam": lambda: Adam(parameters, lr=lr, betas=betas),
        "OAdam": lambda: OAdam(parameters, lr=lr, betas=betas)
    }

    return optimizer_dict[optimizer]()

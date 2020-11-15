from torch.optim import Adam


def create_optimizer(optimizer, parameters, lr, betas):
    optimizer_dict = {
        "Adam": lambda: Adam(parameters, lr=lr, betas=betas)
    }

    return optimizer_dict[optimizer]()

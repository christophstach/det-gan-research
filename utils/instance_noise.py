import torch


def instance_noise(x, global_step, last_global_step):
    # Add instance noise: https://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-training/

    if last_global_step > 0:
        sigma = max((last_global_step - global_step) / last_global_step, 0.0)

        if sigma > 0:
            if isinstance(x, list):
                # msg enabled
                x = [item + torch.randn_like(item) * sigma for item in x]
            else:
                x = x + torch.randn_like(x) * sigma
    else:
        sigma = 0

    return x, sigma
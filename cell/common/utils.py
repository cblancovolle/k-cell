import torch


def clipmin_if_all(x, clip_eps=1e-8):
    if torch.all(x < clip_eps):
        return x.clip(min=clip_eps)
    return x

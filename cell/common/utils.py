import torch


def clipmin_if_all(x, clip_eps=1e-8):
    clip_mask = torch.all(x < clip_eps)
    x_clipped = x.clip(min=clip_eps)
    return torch.where(clip_mask, x_clipped, x)


def clipmin_first_if_all(x, clip_eps=1e-8):
    clip_mask = torch.all(x < clip_eps).view(-1)
    return torch.where(
        (clip_mask & (torch.arange(x.size(0)) == 0)),
        clip_eps,
        x.view(-1),
    ).view(*x.size())

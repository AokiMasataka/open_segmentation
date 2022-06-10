import torch
from builder import OPTIMIZER


__all__ = ['Adam', 'AdamW']


def add_weight_decay(model):
    all_params = set(model.parameters())
    wd_params = set()
    for m in model.modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
            wd_params.add(m.weight)
    no_wd_params = all_params - wd_params
    return list(no_wd_params), list(wd_params)


@OPTIMIZER.register_module
def Adam(model, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0, amsgrad=False):
    no_decay, decay = add_weight_decay(model)

    params = [{'params': no_decay, 'weight_decay': 0.0}, {'params': decay}]

    optimizer = torch.optim.Adam(
        params=params,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        amsgrad=amsgrad,
    )
    return optimizer


@OPTIMIZER.register_module
def AdamW(model, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False):
    no_decay, decay = add_weight_decay(model)

    params = [{'params': no_decay, 'weight_decay': 0.0}, {'params': decay}]

    optimizer = torch.optim.AdamW(
        params=params,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        amsgrad=amsgrad,
    )
    return optimizer

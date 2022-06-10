from torch.optim import lr_scheduler
from builder import OPTIMIZER


@OPTIMIZER.register_module
def CosineAnnealingLR(optimizer, T_max, eta_min=0):
    return lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min, last_epoch=-1)

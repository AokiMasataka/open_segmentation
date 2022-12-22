import torch
from torch import nn
from torch.nn import functional
from ..builder import LOSSES


@LOSSES.register_module
class MSELoss(nn.Module):
    def __init__(self, loss_weight=1.0, class_weight=None):
        super(MSELoss, self).__init__()
        self.loss_weight = loss_weight
        self.class_weight = class_weight
    
    def forward(self, logits, labels):
        if self.class_weight is None:
            loss = functional.mse_loss(input=logits, target=labels)
        else:
            loss = list()
            for weight, logit, label in zip(self.class_weight, logits, labels):
                loss.append(functional.mse_loss(input=logit, target=label) * weight)
            loss = torch.sum(torch.stack(loss, dim=0))
        return loss

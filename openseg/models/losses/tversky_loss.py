from cProfile import label
import torch
from torch import nn
from torch.nn import functional


def binary_tversky_loss(predicts, labels, alpha, beta, smooth, class_weight=None):
    predicts = predicts.view(predicts.shape[0], -1)
    labels = labels.view(labels.shape[0], -1)
    valid_mask = valid_mask.view(valid_mask.shape[0], -1)

    TP = torch.sum(torch.mul(predicts, labels) * valid_mask, dim=1)
    FP = torch.sum(torch.mul(predicts, 1 - labels) * valid_mask, dim=1)
    FN = torch.sum(torch.mul(1 - predicts, labels) * valid_mask, dim=1)
    tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    
    return 1 - tversky


def tversky_loss(predicts, labels, alpha, beta, smooth, class_weight=None):
    loss = list()

    for i in range(predicts.shape[1]):
        class_loss = binary_tversky_loss(
            predicts=predicts[:, i],
            labels=labels[..., i],
            alpha=alpha,
            beta=beta,
            smooth=smooth
        )

        if class_weight is not None:
            class_loss *= class_loss[i]
        
        loss.append(class_loss)
    loss = torch.mean(torch.cat(loss, dim=0), dim=0)
    return loss


class TverskyLoss(nn.Module):
    def __init__(self, mode='binary', alpha=0.3, beta=0.7, smooth=0.0, loss_weight=1.0, weight=None, ignore_index=-100):
        super(TverskyLoss, self).__init__()
        assert mode in ('binary', 'multiclass')
        self.mode = mode
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.loss_weight = loss_weight
        self.weight = weight

        if mode == 'binary':
            self.loss_fn = binary_tversky_loss
        elif mode == 'multiclass':
            self.loss_fn = tversky_loss

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        if self.weight is not None:
            class_weight = logits.new_tensor(self.weight)
        else:
            class_weight = None
        
        predicts = functional.softmax(logits, dim=1)
        loss = self.loss_fn(predicts=predicts, labels=labels, alpha=self.alpha, beta=self.beta, smooth=self.smooth)
        return loss

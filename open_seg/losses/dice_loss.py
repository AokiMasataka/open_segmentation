import numpy as np
import torch
from torch import nn
from torch.nn import functional
from open_seg.builder import LOSSES


BINARY_MODE = 'binary'
MULTICLASS_MODE = 'multiclass'
MULTILABEL_MODE = 'multilabel'


def dice_metric(pred, label, smooth=1.0):
    intersection = np.sum(pred * label)
    cardinality = np.sum(pred + label)
    return (2.0 * intersection + smooth) / (cardinality + smooth).clip(min=1e-6, max=None)


def dice_score(pred, label, smooth=1.0):
    intersection = torch.sum(input=pred * label)
    cardinality = torch.sum(input=pred + label)
    return (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(1e-6)


@LOSSES.register_module
class DiceLoss(nn.Module):
    def __init__(self, mode='binary', smooth=1.0, ignore_index=255, loss_weight=1.0, loss_name='loss_dice'):
        super(DiceLoss, self).__init__()
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        self.mode = mode
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight
        self.loss_name = loss_name

    def forward(self, logits, labels):
        _, num_classes, _, _ = logits.shape
        labels = functional.one_hot(tensor=labels, num_classes=num_classes).float()
        labels = labels.view(-1)
        logits = logits.view(-1)
        assert logits.shape == labels.shape
        preds = logits.sigmoid()

        if self.ignore_index is not None:
            mask = labels != self.ignore_index
            preds = preds * mask
            labels = labels * mask

        score = dice_score(pred=preds, label=labels, smooth=self.smooth)
        loss = (1 - score) * self.loss_weight
        return loss

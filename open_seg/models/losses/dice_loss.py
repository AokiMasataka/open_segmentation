import numpy as np
import torch
from torch import nn
from torch.nn import functional
from ..builder import LOSSES


BINARY_MODE = 'binary'
MULTICLASS_MODE = 'multiclass'


def dice_metric(pred, label, smooth=1.0):
    if 1 == pred.shape[2]:
        pred = pred.squeeze(axis=2)
    elif 1 < pred.shape[2]:
        pred = pred.argmax(axis=2)

    assert label.shape == pred.shape, f'label shape: {label.shape} pred shape: {pred.shape}'
    intersection = np.sum(pred * label)
    cardinality = np.sum(pred + label)
    return (2.0 * intersection + smooth) / (cardinality + smooth).clip(min=1e-6, max=None)


def dice_score(pred, label, smooth=1.0):
    intersection = torch.sum(input=pred * label)
    cardinality = torch.sum(input=pred + label)
    return (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(1e-6)


def multiclass_dice(preds, labels, smooth=1.0, class_weight=None):
    loss = []
    for pred, label in zip(preds, labels):
        loss.append(1 - dice_score(pred=pred, label=label, smooth=smooth))

    loss = torch.stack(loss) * class_weight
    return loss


@LOSSES.register_module
class DiceLoss(nn.Module):
    def __init__(self, mode='binary', smooth=1.0, ignore_index=None, loss_weight=1.0, class_weight=None, loss_name='loss_dice'):
        super(DiceLoss, self).__init__()
        assert mode in {BINARY_MODE, MULTICLASS_MODE}
        self.mode = mode
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.loss_name = loss_name

        if mode == BINARY_MODE:
            self.loss_fn = dice_score
        elif mode == MULTICLASS_MODE:
            self.loss_fn = multiclass_dice

    def forward(self, logits, labels):
        _, num_classes, _, _ = logits.shape

        if self.ignore_index is not None:
            mask = torch.ne(input=labels, other=self.ignore_index).unsqueeze(1)
            logits = logits * mask
            labels = labels * mask

        if self.mode == BINARY_MODE:
            labels = labels.unsqueeze(1)
            assert logits.shape == labels.shape
            score = dice_score(pred=logits.sigmoid(), label=labels, smooth=self.smooth)
            loss = (1 - score) * self.loss_weight
            return loss

        elif self.mode == MULTICLASS_MODE:
            labels = functional.one_hot(tensor=labels, num_classes=num_classes).float()
            assert logits.shape == labels.shape
            loss = multiclass_dice(
                preds=logits.sigmoid(), labels=labels, smooth=self.smooth, class_weight=self.class_weight
            )
            loss = loss * self.loss_weight
            return loss

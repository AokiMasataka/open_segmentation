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



def binary_dice_loss(predicts, labels, smooth=1.0):
    intersection = torch.sum(input=predicts * labels)
    cardinality = torch.sum(input=predicts + labels)
    return 1 - (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(1e-6)


def dice_loss(predicts, labels, smooth=1.0, class_weight=None):
    loss = list()
    for i in range(predicts.shape[1]):
        class_loss = binary_dice_loss(
            predicts=predicts[:, i],
            labels=labels[..., i],
            smooth=smooth
        )
        
        if class_loss is not None:
            class_loss *= class_loss[i]
        
        loss.append(class_loss)
    
    loss = torch.mean(torch.cat(loss, dim=0), dim=0)
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

    def forward(self, logits, labels):
        _, num_classes, _, _ = logits.shape

        if self.ignore_index is not None:
            mask = torch.ne(input=labels, other=self.ignore_index).unsqueeze(1)
            logits = logits * mask
            labels = labels * mask

        if self.mode == BINARY_MODE:
            labels = labels.unsqueeze(1)
            score = binary_dice_loss(pred=logits.sigmoid(), label=labels, smooth=self.smooth)
            loss = (1 - score) * self.loss_weight
            return loss

        elif self.mode == MULTICLASS_MODE:
            labels = functional.one_hot(tensor=labels, num_classes=num_classes).float()
            loss = dice_loss(
                preds=logits.sigmoid(), labels=labels, smooth=self.smooth, class_weight=self.class_weight
            )
            loss = loss * self.loss_weight
            return loss

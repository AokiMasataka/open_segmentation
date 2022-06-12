from typing import Optional
import torch
from torch import nn
from torch.nn import functional
from builder import LOSSES


__all__ = ['SoftBCEWithLogitsLoss']


@LOSSES.register_module
class SoftBCEWithLogitsLoss(nn.Module):
    def __init__(self, loss_weight=1.0, ignore_index=-100, reduction='mean', smooth_factor=None):
        """Drop-in replacement for torch.nn.BCEWithLogitsLoss with few additions: ignore_index and label_smoothing
        Args:
            ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient.
            smooth_factor: Factor to smooth target (e.g. if smooth_factor=0.1 then [1, 0, 1] -> [0.9, 0.1, 0.9])
        Shape
             - **y_pred** - torch.Tensor of shape NxCxHxW
             - **y_true** - torch.Tensor of shape NxHxW or Nx1xHxW
        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        super().__init__()
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.smooth_factor = smooth_factor

    def forward(self, logit, label) -> torch.Tensor:
        """
        Args:
            y_pred: torch.Tensor of shape (N, C, H, W)
            y_true: torch.Tensor of shape (N, H, W)  or (N, 1, H, W)
        Returns:
            loss: torch.Tensor
        """

        if self.smooth_factor is not None:
            soft_targets = (1 - label) * self.smooth_factor + label * (1 - self.smooth_factor)
        else:
            soft_targets = label
        loss = functional.binary_cross_entropy_with_logits(logit, soft_targets, reduction='none')

        if self.ignore_index is not None:
            not_ignored_mask = label != self.ignore_index
            loss *= not_ignored_mask.type_as(loss)

        if self.reduction == 'mean':
            loss = loss.mean()

        if self.reduction == 'sum':
            loss = loss.sum()

        return loss * self.loss_weight

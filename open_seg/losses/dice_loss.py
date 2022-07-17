import numpy as np
import torch
from torch import nn
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
        assert logits.shape == labels.shape
        preds = logits.sigmoid()

        if self.ignore_index is not None:
            mask = labels != self.ignore_index
            preds = preds * mask
            labels = labels * mask

        score = dice_score(pred=preds, label=labels, smooth=self.smooth)
        loss = (1 - score) * self.loss_weight
        return loss



# @LOSSES.register_module
# class DiceLoss(nn.Module):
#     def __init__(
#         self,
#         mode,
#         alpha=0.5,
#         beta=0.5,
#         smooth=1.0,
#         ignore_index=255,
#         eps=1e-7,
#         loss_weight=1.0,
#         loss_name='loss_dice'
#     ):
#         """Implementation of Dice loss for image segmentation task.
#         It supports binary, multiclass and multilabel cases
#
#         Args:
#             mode: Loss mode 'binary', 'multiclass' or 'multilabel'
#
#             log_loss: If True, loss computed as `- log(dice_coeff)`, otherwise `1 - dice_coeff`
#             from_logits: If True, assumes input is raw logits
#             smooth: Smoothness constant for dice coefficient (a)
#             ignore_index: Label that indicates ignored pixels (does not contribute to loss)
#             eps: A small epsilon for numerical stability to avoid zero division error
#                 (denominator will be always greater or equal to eps)
#
#         Shape
#              - **y_pred** - torch.Tensor of shape (N, C, H, W)
#              - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)
#
#         Reference
#             https://github.com/BloodAxe/pytorch-toolbelt
#         """
#         assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
#         super(DiceLoss, self).__init__()
#         self.mode = mode
#
#         self.alpha = alpha
#         self.beta = beta
#         self.smooth = smooth
#         self.eps = eps
#         self.ignore_index = ignore_index
#         self._loss_name = loss_name
#         self.loss_weight = loss_weight
#
#     def forward(self, y_pred, y_true):
#
#         assert y_true.size(0) == y_pred.size(0)
#
#         # Apply activations to get [0..1] class probabilities
#         # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
#         # extreme values 0 and 1
#         if self.mode == MULTICLASS_MODE:
#             y_pred = y_pred.log_softmax(dim=1).exp()
#         else:
#             y_pred = functional.logsigmoid(y_pred).exp()
#
#         bs = y_true.size(0)
#         num_classes = y_pred.size(1)
#         dims = (0, 2)
#
#         if self.mode == BINARY_MODE:
#             y_true = y_true.view(bs, 1, -1)
#             y_pred = y_pred.view(bs, 1, -1)
#
#             if self.ignore_index is not None:
#                 mask = y_true != self.ignore_index
#                 y_pred = y_pred * mask
#                 y_true = y_true * mask
#
#         if self.mode == MULTICLASS_MODE:
#             y_true = y_true.view(bs, -1)
#             y_pred = y_pred.view(bs, num_classes, -1)
#
#             if self.ignore_index is not None:
#                 mask = y_true != self.ignore_index
#                 y_pred = y_pred * mask.unsqueeze(1)
#
#                 y_true = functional.one_hot((y_true * mask).to(torch.long), num_classes)  # N,H*W -> N,H*W, C
#                 y_true = y_true.permute(0, 2, 1) * mask.unsqueeze(1)  # H, C, H*W
#             else:
#                 y_true = functional.one_hot(y_true, num_classes)  # N,H*W -> N,H*W, C
#                 y_true = y_true.permute(0, 2, 1)  # H, C, H*W
#
#         scores = soft_tversky_score(
#             y_pred,
#             y_true.type_as(y_pred),
#             alpha=self.alpha,
#             beta=self.beta,
#             smooth=self.smooth,
#             eps=self.eps,
#             dims=dims
#         )
#
#         loss = 1.0 - scores
#
#         # Dice loss is undefined for non-empty classes
#         # So we zero contribution of channel that does not have true pixels
#         # NOTE: A better workaround would be to use loss term `mean(y_pred)`
#         # for this case, however it will be a modified jaccard loss
#
#         mask = y_true.sum(dims) > 0
#         loss *= mask.to(loss.dtype)
#         return loss.mean() * self.loss_weight

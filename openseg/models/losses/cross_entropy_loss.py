import torch
from torch.nn import functional
from ..builder import LOSSES


def bce_loss(
    predicts: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor = None,
    ignore_index: int = None,
    label_smoothing: float = None
):
    num_classes = predicts.shape[1]
    if num_classes == 1:
        labels = labels.unsqueeze(0)
    else:
        labels = functional.one_hot(labels.long(), num_classes=num_classes).float()
    predicts = predicts.permute(0, 2, 3, 1).contiguous()

    if label_smoothing is not None:
        labels = (1 - labels) * label_smoothing + labels * (1 - label_smoothing)
    
    if ignore_index is not None:
        valid_mask = (labels == ignore_index).long()
        predicts = predicts[valid_mask]
        labels = labels[valid_mask]

    loss = functional.binary_cross_entropy(input=predicts, target=labels, reduction='mean')
    return loss


def ce_loss(
    predicts: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor = None,
    ignore_index: int = -1,
    label_smoothing: float = 0.0
):
    num_classes = predicts.shape[1]
    predicts = predicts.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)
    labels = labels.view(-1)
    loss = functional.cross_entropy(
        input=predicts,
        target=labels.long(),
        weight=weight,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing
    )
    return loss


@LOSSES.register_module
class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, mode='ce', loss_weight: float = 1.0, class_weight: list = None, ignore_index: int = -1):
        super(CrossEntropyLoss, self).__init__()
        assert mode in ('ce', 'bce')
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.ignore_index = ignore_index
        if mode == 'ce':
            self.loss_fn = ce_loss
        elif mode == 'bce':
            self.loss_fn = bce_loss
    
    def forward(self, predicts: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            images: Tensor (batch, dim, w, h)
            labels: Tensor (batch, w, h)
        """
        loss = self.loss_fn(predicts=predicts, labels=labels) * self.loss_weight
        return loss

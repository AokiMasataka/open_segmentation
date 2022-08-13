from torch import nn
from torch.nn import functional
from open_seg.builder import LOSSES


def cross_entropy(logits, labels, weight=None, ignore_index=-100, label_smoothing=0.0):
    assert logits.ndim == labels.ndim - 1
    labels = labels.long()

    loss = functional.cross_entropy(
        input=logits,
        target=labels,
        weight=weight,
        reduction='mean',
        ignore_index=ignore_index,
        label_smoothing=label_smoothing
    )
    return loss


def binary_cross_entropy(logits, labels, weight=None, ignore_index=255, label_smoothing=0.0):
    _, num_classes, _, _ = logits.shape
    labels = functional.one_hot(tensor=labels, num_classes=num_classes)
    assert logits.shape == labels.shape

    labels = (1 - labels) * label_smoothing + labels * (1 - label_smoothing)
    loss = functional.binary_cross_entropy_with_logits(input=logits, target=labels, reduction='mean')
    return loss


@LOSSES.register_module
class CrossEntropyLoss(nn.Module):
    def __init__(self, mode='bce', label_smooth=0.0, loss_weight=1.0, weight=None, ignore_index=-100, loss_name=None):
        super(CrossEntropyLoss, self).__init__()
        assert mode in ('bce', 'ce')
        self.mode = mode
        self.label_smooth = label_smooth
        self.loss_weight = loss_weight
        self.weight = weight
        self.ignore_index = ignore_index

        if mode == 'ce':
            self.loss_fn = cross_entropy
        elif mode == 'bce':
            self.loss_fn = binary_cross_entropy

        if loss_name is None:
            self.loss_name = mode + '_loss'
        else:
            self.loss_name = loss_name

    def forward(self, logits, labels):
        loss = self.loss_fn(
            logits=logits,
            labels=labels,
            weight=self.weight,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smooth
        )
        return loss * self.loss_weight

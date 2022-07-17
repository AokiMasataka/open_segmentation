from torch import nn
from torch.nn import functional
from open_seg.builder import LOSSES


def cross_entropy(logits, labels, weight=None, ignore_index=-100, label_smoothing=0.0):
    assert logits.shape == labels.shape
    labels = labels.argmax(dim=1).long()

    logits = logits.view(-1)
    labels = labels.view(-1)
    loss = functional.cross_entropy(
        input=logits,
        target=labels,
        weight=weight,
        reduction='mean',
        ignore_index=ignore_index,
        label_smoothing=label_smoothing
    )
    return loss


def binary_cross_entropy(logits, labels, label_smoothing=0.0):
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

        if loss_name is None:
            self.loss_name = mode + '_loss'
        else:
            self.loss_name = loss_name

    def forward(self, logits, labels):
        if self.mode == 'ce':
            loss = cross_entropy(
                logits=logits,
                labels=labels,
                weight=self.weight,
                ignore_index=self.ignore_index,
                label_smoothing=self.label_smooth
            )
        else:
            loss = binary_cross_entropy(logits=logits, labels=labels, label_smoothing=self.label_smooth)
        return loss * self.loss_weight

from torch import nn
from torch.nn import functional
from ..builder import LOSSES


def binary_cross_entropy(logits, labels, weight=None, ignore_index=255, label_smoothing=0.0):
    labels = labels.unsqueeze(1)
    labels = (1 - labels) * label_smoothing + labels * (1 - label_smoothing)
    loss = functional.binary_cross_entropy_with_logits(input=logits.sigmoid(), target=labels, reduction='mean')
    return loss


def cross_entropy(logits, labels, weight=None, ignore_index=-100, label_smoothing=0.0):
    b, c, w, h = logits.shape
    logits = logits.permute(0, 2, 3, 1).reshape(b * w * h, c)
    labels = labels.view(b * w * h)

    loss = functional.cross_entropy(
        input=logits,
        target=labels,
        weight=weight,
        reduction='mean',
        ignore_index=ignore_index,
        label_smoothing=label_smoothing
    )
    return loss


@LOSSES.register_module
class CrossEntropyLoss(nn.Module):
    def __init__(self, mode='binary', label_smooth=0.0, loss_weight=1.0, weight=None, ignore_index=-100, loss_name=None):
        super(CrossEntropyLoss, self).__init__()
        assert mode in ('binary', 'multiclass')
        self.mode = mode
        self.label_smooth = label_smooth
        self.loss_weight = loss_weight
        self.weight = weight
        self.ignore_index = ignore_index

        if mode == 'binary':
            self.loss_fn = binary_cross_entropy
        elif mode == 'multiclass':
            self.loss_fn = cross_entropy
        
    def forward(self, logits, labels):
        loss = self.loss_fn(
            logits=logits,
            labels=labels.long(),
            weight=self.weight,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smooth
        )
        return loss * self.loss_weight
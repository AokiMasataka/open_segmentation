from itertools import filterfalse
import numpy as np
import torch
from torch.nn import functional
from open_seg.builder import LOSSES


@LOSSES.register_module
class LovaszHingeLoss(torch.nn.Module):
    def __init__(self, per_image=True, symmetric=False, ignore_index=None, loss_weight=1.0, loss_name='lovasz_hinge'):
        super().__init__()
        self.per_image = per_image
        self.symmetric = symmetric
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight
        self.loss_name = loss_name

        if per_image:
            self.loss_fn = self._per_image
        else:
            self.loss_fn = self._not_per_image

    def forward(self, logits, labels):
        if self.symmetric:
            loss = self.loss_fn(logits=logits, labels=labels) + self.loss_fn(logits=-logits, labels=1.0 - labels)
            loss = loss * 0.5
        else:
            loss = self.loss_fn(logits=logits, labels=labels) * self.loss_weight

        return loss * self.loss_weight

    def _per_image(self, logits, labels):
        loss = [lovasz_hinge(logits=logit, labels=label, ignore_index=self.ignore_index) for logit, label in zip(logits, labels)]
        return sum(loss) / loss.__len__()

    def _not_per_image(self, logits, labels):
        return lovasz_hinge(logits=logits, labels=labels, ignore_index=self.ignore_index)


class DeepSuperVisionLoss(torch.nn.Module):
    def __init__(self, symmetric=True, loss_weight=0.1):
        self.symmetric = symmetric
        self.loss_weight = loss_weight

    def forward(self, logits_deep, label):
        losses = []
        for logit_deep in logits_deep:
            bce_loss = functional.binary_cross_entropy_with_logits(input=logit_deep, target=label)
            if self.symmetric:
                lovasz_loss = self.loss_fn(logits=logits, labels=labels)
                lovasz_loss += self.loss_fn(logits=-logits, labels=1.0 - labels)
                lovasz_loss *= 0.5
            else:
                lovasz_loss = lovasz_hinge(logits=logits, labels=labels, ignore_index=self.ignore_index)
            losses.append((loss + lovasz_loss) * self.loss_weight)
        return sum(losses)


def lovasz_hinge_non_empty(logits_deep, label):
    batch, c, h, w = label.size()
    y2 = label.view(batch * c, -1)
    logits_deep2 = logits_deep.view(batch * c, -1)

    y_sum = torch.sum(y2, dim=1)
    non_empty_idx = (y_sum != 0)

    if non_empty_idx.sum() == 0:
        return torch.tensor(0)
    else:
        loss = functional.binary_cross_entropy_with_logits(logits_deep2[non_empty_idx], y2[non_empty_idx])
        loss += lovasz_hinge(logits_deep2[non_empty_idx].view(-1, h, w), y2[non_empty_idx].view(-1, h, w))
        return loss


def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(dim=0)
    union = gts + (1 - gt_sorted).float().cumsum(dim=0)
    jaccard = 1.0 - intersection / union
    if 1 < p:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge(logits, labels, ignore_index=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      ignore: void class id
    """

    logits = logits.view(-1)
    labels = labels.view(-1)
    if ignore_index is not None:
        valid = (labels != ignore_index)
        logits = logits[valid]
        labels = labels[valid]

    if len(labels) == 0:    # only void pixels, the gradients should be 0
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = (1.0 - logits * signs)
    errors_sorted, perm = torch.sort(input=errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    return torch.dot(functional.relu(errors_sorted), grad)


def f_mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = filterfalse(np.isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n

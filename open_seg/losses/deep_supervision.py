import torch
from torch.nn import functional
from .lovasz_loss import lovasz_hinge


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

import torch
from torch.nn import functional
from ..builder import LOSSES


def iou_score(predicts: torch.Tensor, labels: torch.Tensor, smooth: float = 1.0):
    labels = functional.one_hot(labels, num_classes=predicts.shape[3]).float()
    predicts = predicts.sigmoid()

    numer = torch.sum(predicts * labels) * 2.0
    denor = torch.sum(predicts + labels)
    iou_score = (numer + smooth) / (denor + smooth)
    return iou_score.item()


@LOSSES.register_module
class IOULoss(torch.nn.Module):
    def __init__(self, loss_weight: float = 1.0, class_weight: list = None, ignore_index: int = 255, smooth: float = 1.0):
        super(IOULoss, self).__init__()
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self.smooth = smooth

        if class_weight is not None:
            class_weight = torch.tensor(class_weight, dtype=torch.float)
            self.register_buffer(name='class_weight', tensor=class_weight)
        else:
            self.class_weight = None

    def forward(self, predicts: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            images: Tensor (batch, dim, w, h)
            labels: Tensor (batch, w, h)
        """
        labels = labels.long()
        num_classes = predicts.shape[1]
        
        if self.ignore_index is not None:
            valid_mask = (labels == self.ignore_index) | (labels >= num_classes)
            labels[valid_mask] = 0
            valid_mask = 1 - valid_mask.long()
        
        if num_classes == 1:
            labels = labels.unsqueeze(dim=3)
        else:
            labels = functional.one_hot(labels, num_classes=num_classes)
        predicts = predicts.permute(0, 2, 3, 1)
        predicts = predicts.sigmoid()

        if self.ignore_index is not None:
            labels[:, :, :, 0] = labels[:, :, :, 0] * valid_mask

        numer = torch.sum(torch.mul(predicts, labels), dim=(0, 1, 2)) + self.smooth
        denor = torch.sum(predicts.pow(2) + labels.pow(2), dim=(0, 1, 2)) + self.smooth
        class_dice_score = numer / denor

        class_dice_score = (1.0 - class_dice_score)
        if self.class_weight is not None:
            class_dice_score = class_dice_score * self.class_weight
        return torch.mean(class_dice_score) * self.loss_weight
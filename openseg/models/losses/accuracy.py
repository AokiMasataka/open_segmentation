import numpy as np
import torch


def accuracy(predicts: torch, labels: torch, ignore_index: int = 255):
    """
    Args:
        predicts: Tensor(b, w, h, num_class)
        labels: Tensor(b, w, h)
        ignore_index: int
    """
    predicts = torch.argmax(predicts, dim=3)
    acc = torch.mean((predicts == labels).float()) * 100
    return acc.item()
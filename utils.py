import os
import logging
import numpy
import torch
from torch.backends import cudnn


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def get_logger(log_file=None, stream=None):
    logger = logging.getLogger(__name__)

    if log_file is not None:
        logger.addHandler(logging.FileHandler(filename=log_file))
    if stream is not None:
        logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    return logger
import os
import logging
from typing import Sequence

import numpy
import torch
from torch import nn
from torch.backends import cudnn


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def get_logger(log_file=None, stream=True):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    if stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger


def cast_tensor_dtype(inputs, src_type, dst_type):
    if isinstance(inputs, torch.Tensor):
        return inputs.to(dst_type) if inputs.dtype == src_type else inputs
    elif isinstance(inputs, (list, tuple)):
        return [cast_tensor_dtype(inputs=item, src_type=src_type, dst_type=dst_type) for item in inputs]
    else:
        return inputs


def force_fp16(func):
    def wrapper(*args):
        args = cast_tensor_dtype(inputs=args, src_type=torch.float, dst_type=torch.half)
        ret = func(*args)
        return cast_tensor_dtype(inputs=ret, src_type=torch.half, dst_type=torch.float)
    return wrapper


def force_fp32(func):
    def wrapper(*args):
        args = cast_tensor_dtype(inputs=args, src_type=torch.half, dst_type=torch.float)
        ret = func(*args)
        return cast_tensor_dtype(inputs=ret, src_type=torch.float, dst_type=torch.half)
    return wrapper


def conv3x3(in_channel, out_channel):
    return nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=1, dilation=(1, 1), bias=False)


def conv1x1(in_channel, out_channel):
    return nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=(1, 1), padding=0, dilation=(1, 1), bias=False)


def init_weight(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Batch') != -1:
        m.weight.data.normal_(1, 0.02)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Embedding') != -1:
        nn.init.orthogonal_(m.weight, gain=1)


def to_tuple(param, low=None, bias=None):
    if low is not None and bias is not None:
        raise ValueError("Arguments low and bias are mutually exclusive")

    if param is None:
        return param

    if isinstance(param, (int, float)):
        if low is None:
            param = -param, +param
        else:
            param = (low, param) if low < param else (param, low)
    elif isinstance(param, Sequence):
        param = tuple(param)
    else:
        raise ValueError("Argument param must be either scalar (int, float) or tuple")

    if bias is not None:
        return tuple(bias + x for x in param)

    return tuple(param)

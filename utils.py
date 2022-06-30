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

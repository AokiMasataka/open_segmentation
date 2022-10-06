import torch


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

import copy
from typing import Sequence
import cv2
import numpy as np
import torch
from ..builder import PIPELINES

try:
    import albumentations
except:
    pass



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


@PIPELINES.register_module
class Identity:
    def __init__(self):
        pass

    def __call__(self, result: dict):
        return result


@PIPELINES.register_module
class Compose:
    def __init__(self, transforms: dict):
        self.transforms = transforms

    def __call__(self, result: dict):
        for transform in self.transforms.values():
            result = transform(result)

        if isinstance(result.get('image', None), np.ndarray):
            result['image'] = np.ascontiguousarray(result['image'])
            if 'label' in result:
                result['label'] = np.ascontiguousarray(result['label'])

        return result
    
    def __getitem__(self, item: str):
        return self.transforms.get(item, None)


@PIPELINES.register_module
class ToTensor:
    def __init__(self):
        pass

    def __call__(self, result: dict):
        # transpose
        result['image'] = result['image'].transpose(2, 0, 1)
        # to tensor
        result['image'] = torch.tensor(result['image'], dtype=torch.float)
        if 'label' in result:
            result['label'] = torch.tensor(result['label'], dtype=torch.long)
        return result


@PIPELINES.register_module
class Resize:
    def __init__(self, size: int, keep_retio: bool = True, ignore_label: bool = False):
        self.new_size = size
        self.keep_retio = keep_retio
        self.ignore_label = ignore_label

    def __call__(self, result: dict):
        if self.keep_retio:
            result['image'] = self._reisze_retio(result['image'])
            if 'label' in result and not self.ignore_label:
                result['label'] = self._reisze_retio(result['label'])
        else:
            result['image'] = cv2.resize(result['image'], dsize=(self.new_size, self.new_size))
            if 'label' in result and not self.ignore_label:
                result['label'] = cv2.resize(result['label'], dsize=(self.new_size, self.new_size))

        return result

    def _reisze_retio(self, image):
        height, width, = image.shape[:2]
        retio = self.new_size / max(height, width)

        new_h = round(height * retio)
        new_w = round(width * retio)
        image = cv2.resize(image, dsize=(new_w, new_h))
        return image


@PIPELINES.register_module
class Padding:
    def __init__(self, size: int = None, pad_value: int = 0, label_pad_value: int = 0):
        self.size = size
        self.pad_value = pad_value
        self.label_pad_value = label_pad_value

    def __call__(self, result: dict):
        height, width, ch = result['image'].shape

        if self.size == height == width:
            result['pad_t'], result['pad_b'], result['pad_l'], result['pad_r'] = 0, 0, 0, 0
        else:
            top, bottom, left, right = self._create_pad(height=height, width=width)
            result['pad_t'] = top
            result['pad_b'] = bottom
            result['pad_l'] = left
            result['pad_r'] = right
            result['image'] = np.pad(
                array=result['image'],
                pad_width=((top, bottom), (left, right), (0, 0)),
                mode='constant',
                constant_values=self.pad_value
            )

            if 'label' in result:
                result['label'] = np.pad(
                    array=result['label'],
                    pad_width=((top, bottom), (left, right)),
                    mode='constant',
                    constant_values=self.label_pad_value
                )

        return result

    def _create_pad(self, height, width):
        size = max(width, height) if self.size is None else self.size
        width_buff = size - width
        height_buff = size - height

        top_pad = height_buff // 2
        bottom_pad = top_pad + height_buff % 2

        left_pad = width_buff // 2
        right_pad = left_pad + width_buff % 2
        return top_pad, bottom_pad, left_pad, right_pad


@PIPELINES.register_module
class RemovePad:
    def __init__(self):
        pass

    def __call__(self, result: dict):
        w, h = result['image'].shape[:2]
        pad_b = h - result['pad_b']
        pad_r = w - result['pad_r']
        result['image'] = result['image'][result['pad_t']:pad_b, result['pad_l']:pad_r]
        if 'label' in result:
            result['label'] = result['label'][result['pad_t']:pad_b, result['pad_l']:pad_r]
        return result


@PIPELINES.register_module
class RandomCrop:
    def __init__(self, crop_size: int):
        self.crop_size = crop_size

    def __call__(self, result: dict):
        size = result['image'].shape[0]
        crop_x, crop_y = np.random.randint(low=0, high=size - self.crop_size, size=2, dtype=np.int32)

        result['image'] = result['image'][crop_x: crop_x + self.crop_size, crop_y: crop_y + self.crop_size, :]
        if 'label' in result:
            result['label'] = result['label'][crop_x: crop_x + self.crop_size, crop_y: crop_y + self.crop_size]
        return result


@PIPELINES.register_module
class RandomResizeCrop:
    def __init__(self, min_crop_size: int, max_crop_size: int, size: int):
        self.min_crop_size = min_crop_size
        self.max_crop_size = max_crop_size
        self.size = size

    def __call__(self, result: dict):
        size = result['image'].shape[0]
        crop_size = np.random.randint(low=self.min_crop_size, high=self.max_crop_size)
        crop_x, crop_y = np.random.randint(low=0, high=size - crop_size, size=2, dtype=np.int32)

        result['image'] = result['image'][crop_x: crop_x + crop_size, crop_y: crop_y + crop_size, :]
        result['image'] = cv2.resize(result['image'], dsize=(self.size, self.size))
        if 'label' in result:
            result['label'] = result['label'][crop_x: crop_x + crop_size, crop_y: crop_y + crop_size]
            result['label'] = cv2.resize(result['label'], dsize=(self.size, self.size))

        return result


@PIPELINES.register_module
class RandomFlipHorizontal:
    def __init__(self, prob: float = 0.5):
        assert 0.0 <= prob <= 1.0
        self.prob = prob
        self.FLIP_HIRIZONTAL = 1

    def __call__(self, result: dict):
        result['h_flip'] = True
        if np.random.random() < self.prob:
            result['image'] = result['image'][:, ::-1]
            if 'label' in result:
                result['label'] = result['label'][:, ::-1]
        return result


@PIPELINES.register_module
class RandomFlipVertical:
    def __init__(self, prob: float = 0.5):
        assert 0.0 <= prob <= 1.0
        self.prob = prob
        self.FLIP_VERTICAL = 0

    def __call__(self, result: dict):
        result['b_flip'] = False
        if np.random.random() < self.prob:
            result['image'] = result['image'][::-1, :]
            if 'label' in result:
                result['label'] = result['label'][::-1, :]
        return result


@PIPELINES.register_module
class ShiftScaleRotateShear:
    # TODO interpolation args into str
    def __init__(
        self,
        shift_limit: float = 0.0625,
        scale_limit: float = 0.1,
        rotate_limit: float = 45.0,
        interpolation:str = 'linear',
        border_mode: str = 'refrect',
        shift_limit_x: float = None,
        shift_limit_y: float = None,
        shear: float = 0.0,
        pad_value: int = 0,
        label_pad_value: int = 255,
        prob: float = 0.5
    ):
        inter = {'linear': cv2.INTER_LINEAR, 'cubic': cv2.INTER_CUBIC, 'lanc': cv2.INTER_LANCZOS4}
        border = {'refrect': cv2.BORDER_REFLECT_101}
        assert interpolation in inter.keys()
        assert border_mode in border.keys()
        self.shift_limit_x = to_tuple(shift_limit_x if shift_limit_x is not None else shift_limit)
        self.shift_limit_y = to_tuple(shift_limit_y if shift_limit_y is not None else shift_limit)
        self.scale_limit = to_tuple(scale_limit, bias=1.0)
        self.rotate_limit = to_tuple(rotate_limit)
        self.interpolation = inter[interpolation]
        self.border_mode = border[border_mode]
        self.shear = shear
        self.pad_value = pad_value
        self.label_pad_value = label_pad_value
        self.prob = prob
        
        
        

    def __call__(self, result: dict):
        angle = np.random.uniform(self.rotate_limit[0], self.rotate_limit[1])
        scale = np.random.uniform(self.scale_limit[0], self.scale_limit[1])
        shfit_x = np.random.uniform(self.shift_limit_x[0], self.shift_limit_x[1])
        shift_y = np.random.uniform(self.shift_limit_y[0], self.shift_limit_y[1])
        shear_x = np.random.uniform(-self.shear, self.shear)
        shear_y = np.random.uniform(-self.shear, self.shear)

        image_shape = result['image'].shape
        _matrix = self._get_matrix(
            image_shape=image_shape,
            angle=angle,
            scale=scale,
            shfit_x=shfit_x,
            shfit_y=shift_y,
            shear_x=shear_x,
            shear_y=shear_y
        )

        result['image'] = cv2.warpAffine(
            src=result['image'],
            M=_matrix,
            dsize=(image_shape[1], image_shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=self.pad_value
        )
        if 'label' in result:
            result['label'] = cv2.warpAffine(
                src=result['label'],
                M=_matrix,
                dsize=(image_shape[1], image_shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=self.label_pad_value
            )

        return result

    @staticmethod
    def _get_matrix(image_shape, angle, scale, shfit_x, shfit_y, shear_x, shear_y):
        height, width = image_shape[:2]
        center = (width / 2, height / 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        rot_matrix[0, 2] += shfit_x * width
        rot_matrix[1, 2] += shfit_y * height

        shear_matrix = np.eye(3, 3)
        shear_matrix[0, 1] = shear_x
        shear_matrix[1, 0] = shear_y

        return rot_matrix @ shear_matrix


@PIPELINES.register_module
class Album:
    """Albumentation augmentation.
    Adds custom transformations from Albumentations library.
    Please, visit `https://albumentations.readthedocs.io`
    to get more information.
    An example of ``transforms`` is as followed:
    .. code-block::
        [
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.0625,
                scale_limit=0.0,
                rotate_limit=0,
                interpolation=1,
                p=0.5),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=[0.1, 0.3],
                contrast_limit=[0.1, 0.3],
                p=0.2),
            dict(type='ChannelShuffle', p=0.1),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Blur', blur_limit=3, p=1.0),
                    dict(type='MedianBlur', blur_limit=3, p=1.0)
                ],
                p=0.1),
        ]
    """

    def __init__(self, transforms):
        transforms = copy.deepcopy(transforms)
        self.transforms = transforms
        self.aug = albumentations.Compose([self.albu_builder(t) for t in self.transforms])

    def albu_builder(self, cfg):
        """Import a module from albumentations.
        It inherits some of :func:`build_from_cfg` logic.
        Args:
            cfg (dict): Config dict. It should at least contain the key "type".
        Returns:
            obj: The constructed object.
        """

        assert isinstance(cfg, dict) and 'type' in cfg
        args = cfg.copy()

        obj_type = args.pop('type')
        if type(obj_type) == str:
            obj_cls = getattr(albumentations, obj_type)
        else:
            raise TypeError(f'type must be a str or valid type, but got {type(obj_type)}')

        if 'transforms' in args:
            args['transforms'] = [self.albu_builder(transform) for transform in args['transforms']]

        return obj_cls(**args)

    def __call__(self, result: dict):
        augged = self.aug(image=result['image'], mask=result['label'])
        result['image'] = augged['image']
        result['label'] = augged['mask']
        return result

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(transforms={self.transforms})'
        return repr_str
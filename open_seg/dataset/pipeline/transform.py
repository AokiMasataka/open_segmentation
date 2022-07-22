import copy
import cv2
import numpy as np
import torch

try:
    import albumentations
except:
    pass

from open_seg.utils import to_tuple
from open_seg.builder import PIPELINES

__all__ = [
    'Compose',
    'Resize',
    'Padding',
    'RemovePad',
    'RandomFlipHorizontal',
    'FlipHorizontal',
    'RandomFlipVertical',
    'FlipVertical',
    'ShiftScaleRotateShear',
]


@PIPELINES.register_module
class Compose:
    def __init__(self, transforms: dict):
        self.transforms = transforms

    def __call__(self, results):
        for transform in self.transforms.values():
            results = transform(results)

        if isinstance(results['image'], np.ndarray):
            results['image'] = np.ascontiguousarray(results['image'])
            if 'label' in results:
                results['label'] = np.ascontiguousarray(results['label'])

            results = self._transpose(results)
            results = self._to_tensor(results)
        return results

    @staticmethod
    def _transpose(results):
        results['image'] = results['image'].transpose(2, 0, 1)
        if 'label' in results:
            results['label'] = results['label'].transpose(2, 0, 1)
        return results

    @staticmethod
    def _to_tensor(results):
        results['image'] = torch.tensor(results['image'], dtype=torch.float)
        if 'label' in results:
            results['label'] = torch.tensor(results['label'], dtype=torch.float)
        return results


@PIPELINES.register_module
class Resize:
    def __init__(self, size, keep_retio=True, interpolation=''):
        self.new_size = size
        self.keep_retio = keep_retio
        self.interpolation = interpolation

    def __call__(self, results):
        if self.keep_retio:
            results['image'] = self._reisze_retio(results['image'])
            if 'label' in results:
                results['label'] = self._reisze_retio(results['label'])
        else:
            results['image'] = cv2.resize(results['image'], dsize=(self.new_size, self.new_size))
            if 'label' in results:
                results['label'] = cv2.resize(results['label'], dsize=(self.new_size, self.new_size))
                if results['label'].ndim == 2:
                    results['label'] = np.expand_dims(a=results['label'], axis=2)
        return results

    def _reisze_retio(self, image):
        height, width, = image.shape[:2]
        retio = self.new_size / max(height, width)

        new_h = int(height * retio)
        new_w = int(width * retio)
        image = cv2.resize(image, dsize=(new_w, new_h))

        if image.ndim == 2:
            image = np.expand_dims(a=image, axis=2)
        return image


@PIPELINES.register_module
class Padding:
    def __init__(self, size=None, pad_value=0, label_pad_value=0):
        self.size = size
        self.pad_value = pad_value
        self.label_pad_value = label_pad_value

    def __call__(self, results):
        height, width, ch = results['image'].shape

        if self.size == height == width:
            results['pad_t'] = 0
            results['pad_b'] = 0
            results['pad_l'] = 0
            results['pad_r'] = 0
            return results

        top, bottom, left, right = self._create_pad(height=height, width=width)
        results['pad_t'] = top
        results['pad_b'] = bottom
        results['pad_l'] = left
        results['pad_r'] = right

        results['image'] = np.pad(
            array=results['image'],
            pad_width=((top, bottom), (left, right), (0, 0)),
            mode='constant',
            constant_values=self.pad_value
        )

        if 'label' in results:
            results['label'] = np.pad(
                array=results['label'],
                pad_width=((top, bottom), (left, right), (0, 0)),
                mode='constant',
                constant_values=self.label_pad_value
            )

        return results

    def _create_pad(self, height, width):
        size = max(width, height) if self.size is None else self.size
        width_buff = size - width
        height_buff = size - height

        top_pad = height_buff // 2
        bottom_pad = top_pad + height_buff % 2

        left_pad = width_buff // 2
        right_pad = left_pad + width_buff % 2
        return top_pad, bottom_pad, left_pad, right_pad


class RemovePad:
    def __init__(self):
        pass

    def __call__(self, results):
        w, h = results['image'].shape[:2]
        pad_b = h - results['pad_b']
        pad_r = w - results['pad_r']
        results['image'] = results['image'][results['pad_t']:pad_b, results['pad_l']:pad_r, :]
        if 'label' in results:
            results['label'] = results['label'][results['pad_t']:pad_b, results['pad_l']:pad_r, :]
        return results


@PIPELINES.register_module
class RandomCrop:
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, results):
        size = results['image'].shape[0]
        crop_x, crop_y = np.random.randint(low=0, high=size - self.crop_size, size=2, dtype=np.int32)

        results['image'] = results['image'][crop_x: crop_x + self.crop_size, crop_y: crop_y + self.crop_size, :]
        if 'label' in results:
            results['label'] = results['label'][crop_x: crop_x + self.crop_size, crop_y: crop_y + self.crop_size, :]
        return results


@PIPELINES.register_module
class RandomResizeCrop:
    def __init__(self, min_crop_size, max_crop_size, size):
        self.min_crop_size = min_crop_size
        self.max_crop_size = max_crop_size
        self.size = size

    def __call__(self, results):
        size = results['image'].shape[0]
        crop_size = np.random.randint(low=self.min_crop_size, high=self.max_crop_size)
        crop_x, crop_y = np.random.randint(low=0, high=size - crop_size, size=2, dtype=np.int32)

        results['image'] = results['image'][crop_x: crop_x + crop_size, crop_y: crop_y + crop_size, :]
        results['image'] = cv2.resize(results['image'], dsize=(self.size, self.size))
        if 'label' in results:
            results['label'] = results['label'][crop_x: crop_x + crop_size, crop_y: crop_y + crop_size, :]
            results['label'] = cv2.resize(results['label'], dsize=(self.size, self.size))
            if results['label'].ndim == 2:
                results['label'] = np.expand_dims(a=results['label'], axis=2)
        return results


@PIPELINES.register_module
class FlipHorizontal:
    def __init__(self):
        self.FLIP_HIRIZONTAL = 1

    def __call__(self, results):
        results['image'] = results['image'][:, ::-1]
        if 'label' in results:
            results['label'] = results['label'][:, ::-1]
        return results


@PIPELINES.register_module
class FlipVertical:
    def __init__(self):
        self.FLIP_VERTICAL = 0

    def __call__(self, results):
        results['image'] = results['image'][::-1, :]
        if 'label' in results:
            results['label'] = results['label'][::-1, :]
        return results


@PIPELINES.register_module
class RandomFlipHorizontal:
    def __init__(self, prob=0.5):
        assert 0.0 <= prob <= 1.0
        self.prob = prob
        self.FLIP_HIRIZONTAL = 1

    def __call__(self, results):
        if np.random.random() < self.prob:
            results['image'] = results['image'][:, ::-1]
            if 'label' in results:
                results['label'] = results['label'][:, ::-1]
        return results


@PIPELINES.register_module
class RandomFlipVertical:
    def __init__(self, prob=0.5):
        assert 0.0 <= prob <= 1.0
        self.prob = prob
        self.FLIP_VERTICAL = 0

    def __call__(self, results):
        if np.random.random() < self.prob:
            results['image'] = results['image'][::-1, :]
            if 'label' in results:
                results['label'] = results['label'][::-1, :]

        return results


@PIPELINES.register_module
class ShiftScaleRotateShear:
    def __init__(
            self,
            shift_limit=0.0625,
            scale_limit=0.1,
            rotate_limit=45,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_REFLECT_101,
            value=None,
            mask_value=None,
            shift_limit_x=None,
            shift_limit_y=None,
            shear=0.0,
            prob=0.5
    ):
        self.shift_limit_x = to_tuple(shift_limit_x if shift_limit_x is not None else shift_limit)
        self.shift_limit_y = to_tuple(shift_limit_y if shift_limit_y is not None else shift_limit)
        self.scale_limit = to_tuple(scale_limit, bias=1.0)
        self.rotate_limit = to_tuple(rotate_limit)
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value
        self.shear = shear
        self.prob = prob

    def __call__(self, results):
        angle = np.random.uniform(self.rotate_limit[0], self.rotate_limit[1])
        scale = np.random.uniform(self.scale_limit[0], self.scale_limit[1])
        shfit_x = np.random.uniform(self.shift_limit_x[0], self.shift_limit_x[1])
        shift_y = np.random.uniform(self.shift_limit_y[0], self.shift_limit_y[1])
        shear_x = np.random.uniform(-self.shear, self.shear)
        shear_y = np.random.uniform(-self.shear, self.shear)

        image_shape = results['image'].shape
        _matrix = self._get_matrix(
            image_shape=image_shape,
            angle=angle,
            scale=scale,
            shfit_x=shfit_x,
            shfit_y=shift_y,
            shear_x=shear_x,
            shear_y=shear_y
        )

        results['image'] = cv2.warpAffine(
            src=results['image'],
            M=_matrix,
            dsize=(image_shape[1], image_shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        if 'label' in results:
            results['label'] = cv2.warpAffine(
                src=results['label'],
                M=_matrix,
                dsize=(image_shape[1], image_shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )

        return results

    def _get_matrix(self, image_shape, angle, scale, shfit_x, shfit_y, shear_x, shear_y):
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

        # Args will be modified later, copying it will be safer
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

    def __call__(self, results):
        augged = self.aug(image=results['image'], mask=results['label'])
        results['image'] = augged['image']
        results['label'] = augged['mask']
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(transforms={self.transforms})'
        return repr_str

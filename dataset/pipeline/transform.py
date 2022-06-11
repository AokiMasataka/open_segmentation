import copy
import math
import cv2
import numpy as np
import torch

import albumentations

from builder import PIPELINES

__all__ = [
    'Compose',
    'LoadImageFromFile',
    'LoadAnnotations',
    'Resize',
    'Padding',
    'RandomFlipHorizontal',
    'FlipHorizontal',
    'RandomFlipVertical',
    'FlipVertical'
]


@PIPELINES.register_module
class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, results):
        for transform in self.transforms:
            results = transform(results)

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
class LoadImageFromFile:
    def __init__(self, to_float=False, color_type='color', max_value=None, force_3chan=False):
        self.to_float = to_float
        self.color_type = color_type
        self.max_value = max_value
        self.force_3chan = force_3chan

    def __call__(self, results):
        image_file = results['image_path']

        if self.color_type == 'color':
            image = cv2.cvtColor(cv2.imread(image_file, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        elif self.color_type == 'anydepth':
            image = cv2.imread(image_file, cv2.IMREAD_ANYDEPTH)
        else:
            Warning('color_type is color or anydepth')
            image = cv2.cvtColor(cv2.imread(image_file, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        if self.to_float:
            try:
                image = image.astype(np.float32)
            except:
                print('None type object file:', image_file)
                assert image == np.ndarray

        if self.max_value == 'max':
            image = image / np.max(image)

        if self.force_3chan:
            image = np.stack([image for _ in range(3)], -1)

        results['image'] = image
        results['original_shape'] = (image.shape[1], image.shape[0])
        results['scale_factor'] = 1.0
        return results


@PIPELINES.register_module
class LoadAnnotations:
    def __init__(self):
        pass

    def __call__(self, results):
        label_file = results['label_path']
        label = cv2.imread(label_file, cv2.IMREAD_COLOR)
        results['label'] = label
        return results


@PIPELINES.register_module
class Resize:
    def __init__(self, size, keep_retio=True):
        self.new_size = size
        self.keep_retio = keep_retio

    def __call__(self, results):
        if self.keep_retio:
            results['image'] = self._reisze_retio(results['image'])
            if 'label' in results:
                results['label'] = self._reisze_retio(results['label'])
        else:
            results['image'] = cv2.resize(results['image'], dsize=self.new_size)
            if 'label' in results:
                results['label'] = cv2.resize(results['label'], dsize=self.new_size)
        return results

    def _reisze_retio(self, image):
        height, width = image.shape[:2]
        retio = self.new_size / max(height, width)

        new_h = int(height * retio)
        new_w = int(width * retio)
        return cv2.resize(image, dsize=(new_w, new_h))


@PIPELINES.register_module
class Padding:
    def __init__(self, size=None, pad_value=0, label_pad_value=0):
        self.size = size
        self.pad_value = pad_value
        self.label_pad_value = label_pad_value

    def __call__(self, results):
        height, width, ch = results['image'].shape

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
                constant_values=self.pad_value
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


@PIPELINES.register_module
class FlipHorizontal:
    def __init__(self):
        self.FLIP_HIRIZONTAL = 1

    def __call__(self, results):
        results['image'] = cv2.flip(results['image'], self.FLIP_HIRIZONTAL)
        if 'label' in results:
            results['label'] = cv2.flip(results['label'], self.FLIP_HIRIZONTAL)
        return results


@PIPELINES.register_module
class FlipVertical:
    def __init__(self):
        self.FLIP_VERTICAL = 0

    def __call__(self, results):
        results['image'] = cv2.flip(results['image'], self.FLIP_VERTICAL)
        if 'label' in results:
            results['label'] = cv2.flip(results['label'], self.FLIP_VERTICAL)
        return results


@PIPELINES.register_module
class RandomFlipHorizontal:
    def __init__(self, prob=0.5):
        assert 0.0 <= prob <= 1.0
        self.prob = prob
        self.FLIP_HIRIZONTAL = 1

    def __call__(self, results):
        if np.random.random() < self.prob:
            results['image'] = cv2.flip(results['image'], self.FLIP_HIRIZONTAL)
            if 'label' in results:
                results['label'] = cv2.flip(results['label'], self.FLIP_HIRIZONTAL)

        return results


@PIPELINES.register_module
class RandomFlipVertical:
    def __init__(self, prob=0.5):
        assert 0.0 <= prob <= 1.0
        self.prob = prob
        self.FLIP_VERTICAL = 0

    def __call__(self, results):
        if np.random.random() < self.prob:
            results['image'] = cv2.flip(results['image'], self.FLIP_VERTICAL)
            if 'label' in results:
                results['label'] = cv2.flip(results['label'], self.FLIP_VERTICAL)

        return results


@PIPELINES.register_module
class RandomPerspective:
    def __init__(self, degrees=0, translate=0.0, scale=0.0, shear=1, perspective=0.0, center=4):
        assert 0.0 <= translate <= 1.0
        assert 0.0 <= scale <= 1.0
        self.degrees = degrees
        self.translate = translate
        if not isinstance(scale, (list, tuple)):
            self.scale = (scale, scale)
        self.shear = shear
        self.perspective = perspective
        self.center = center

    def __call__(self, results):
        shape = results['image'].shape
        matrix = self._get_affine_matrix(shape)

        if self.perspective:
            results['image'] = cv2.warpPerspective(
                results['image'], matrix, dsize=(shape[1], shape[0]), borderValue=(0, 0, 0)
            )

            if 'label' in results:
                results['label'] = cv2.warpPerspective(
                    results['label'], matrix, dsize=(shape[1], shape[0]), borderValue=(0, 0, 0)
                )

        else:
            results['image'] = cv2.warpAffine(
                results['image'], matrix[:2], dsize=(shape[1], shape[0]), borderValue=(0, 0, 0)
            )

            if 'label' in results:
                results['label'] = cv2.warpAffine(
                    results['label'], matrix, dsize=(shape[1], shape[0]), borderValue=(0, 0, 0)
                )
        return results

    def _get_affine_matrix(self, image_shape):
        center = np.eye(3, 3)
        center[0, 2] = -image_shape[1] / center
        center[1, 2] = -image_shape[0] / center

        perspective = np.eye(3, 3)
        perspective[2, 0] = np.random.uniform(-perspective, perspective)
        perspective[2, 1] = np.random.uniform(-perspective, perspective)

        rotation = np.eye(3, 3)
        angle = np.random.uniform(-self.degrees, self.degrees)
        scale = np.random.uniform(1 - self.scale[0], 1 + self.scale[1])
        rotation[:2] = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale)

        shear = np.eye(3, 3)
        shear[0, 1] = math.tan(np.random.uniform(-shear, shear) * math.pi / 180)
        shear[1, 0] = math.tan(np.random.uniform(-shear, shear) * math.pi / 180)

        translation = np.eye(3, 3)
        translation[0, 2] = np.random.uniform(0.5 - self.translate, 0.5 + self.translate) * image_shape[1]
        translation[1, 2] = np.random.uniform(0.5 - self.translate, 0.5 + self.translate) * image_shape[0]

        # order of operations (right to left) is IMPORTANT
        matrix = center @ perspective @ rotation @ shear @ translation
        return matrix


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

    @staticmethod
    def mapper(d, keymap):
        """Dictionary mapper. Renames keys according to keymap provided.

        Args:
            d (dict): old dict
            keymap (dict): {'old_key':'new_key'}
        Returns:
            dict: new dict.
        """

        updated_dict = {}
        for k, v in zip(d.keys(), d.values()):
            new_k = keymap.get(k, k)
            updated_dict[new_k] = d[k]
        return updated_dict

    def a__call__(self, results):
        # dict to albumentations format
        res = copy.deepcopy(results)
        seg_keys = results.get('seg_fields', [])
        masks = []
        for key in seg_keys:
            masks.append(results[key])

        augged = self.aug(image=results['img'], masks=masks)
        results['img'] = augged['image']
        for i, key in enumerate(seg_keys):
            results[key] = augged['masks'][i]

        return results

    def __call__(self, results):
        augged = self.aug(image=results['image'], mask=results['label'])
        results['image'] = augged['image']
        results['label'] = augged['mask']
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(transforms={self.transforms})'
        return repr_str

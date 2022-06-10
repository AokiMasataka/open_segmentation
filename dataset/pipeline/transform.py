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
    'RandomFlipVertical'
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
        if height < width:
            retio = self.new_size / width
        else:
            retio = self.new_size / height

        new_h = int(height * retio)
        new_w = int(width * retio)
        return cv2.resize(image, dsize=(new_w, new_h))


@PIPELINES.register_module
class Padding:
    def __init__(self, pad_value=0, label_pad_value=0):
        self.pad_value = pad_value
        self.label_pad_value = label_pad_value

    def __call__(self, results):
        height, width = results['image'].shape[:2]

        if height < width:
            buff = width - height
            pad_array_top = np.zeros((int(buff // 2), width, 3))
            pad_array_bottom = np.zeros((int(buff // 2 + buff % 2), width, 3))
            results['image'] = np.vstack((
                pad_array_top + self.label_pad_value,
                results['image'],
                pad_array_bottom + self.label_pad_value
            ))
            if 'label' in results:
                results['label'] = np.vstack((
                    pad_array_top + self.label_pad_value,
                    results['label'],
                    pad_array_bottom + self.label_pad_value
                ))

            results['pad_t'] = int(buff // 2)
            results['pad_b'] = int(buff // 2 + buff % 2)
            results['pad_l'] = 0
            results['pad_r'] = 0
        elif width < height:
            buff = height - width
            pad_array_l = np.zeros((height, int(buff // 2), 3))
            pad_array_r = np.zeros((int(buff // 2 + buff % 2), height, 3))
            results['image'] = np.hstack((
                pad_array_l + self.label_pad_value,
                results['image'],
                pad_array_r + self.label_pad_value
            ))
            if 'label' in results:
                results['laebl'] = np.hstack((
                    pad_array_l + self.label_pad_value,
                    results['laebl'],
                    pad_array_r + self.label_pad_value
                ))

            results['pad_t'] = 0
            results['pad_b'] = 0
            results['pad_l'] = int(buff // 2)
            results['pad_r'] = int(buff // 2 + buff % 2)
        else:
            results['pad_t'] = 0
            results['pad_b'] = 0
            results['pad_l'] = 0
            results['pad_r'] = 0

        return results


@PIPELINES.register_module
class RandomFlip:
    def __init__(self, prob=0.5, lr=True, ub=False):
        self.prob = prob
        self.lr = lr
        self.ub = ub

    def __call__(self, results):
        if np.random.random() < self.prob:
            results['image'] = results['image'][:, ::-1, :]
            if 'label' in results:
                results['label'] = results['label'][:, ::-1, :]

        return results


@PIPELINES.register_module
class RandomFlipHorizontal:
    def __init__(self, prob=0.0):
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
    def __init__(self, prob=0.0):
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
    def __init__(self, degrees=0, translate=0.0, scale=0.0, shear=1, perspective=0.0):
        assert 0.0 <= translate <= 1.0
        assert 0.0 <= scale <= 1.0
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective

    def __call__(self, results):
        return results

    def _get_affine_matrix(self):
        affine_matrix = None
        return affine_matrix


def random_perspective(im,
                       targets=(),
                       segments=(),
                       degrees=10,
                       translate=.1,
                       scale=.1,
                       shear=10,
                       perspective=0.0,
                       border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = im.shape[0] + border[0] * 2  # shape(h,w,c)
    width = im.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(im[:, :, ::-1])  # base
    # ax[1].imshow(im2[:, :, ::-1])  # warped

    # Transform label coordinates
    n = len(targets)
    if n:
        use_segments = any(x.any() for x in segments)
        new = np.zeros((n, 4))
        if use_segments:  # warp segments
            segments = resample_segments(segments)  # upsample
            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T  # transform
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # perspective rescale or affine

                # clip
                new[i] = segment2box(xy, width, height)

        else:  # warp boxes
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return im, targets


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

import cv2
from PIL import Image
import numpy as np

from open_seg.builder import PIPELINES


__all__ = [
    'LoadImageFromFile',
    'LoadNumpyImage',
    'LoadAnnotations',
]


@PIPELINES.register_module
class LoadImageFromFile:
    def __init__(self, to_float=False, color_type='color', max_value=None, force_3chan=False):
        self.to_float = to_float
        self.color_type = color_type
        self.max_value = max_value
        self.force_3chan = force_3chan

        assert color_type in ['color', 'anydepth', 'unchanged']

    def __call__(self, results):
        image_file = results['image_path']

        if self.color_type == 'color':
            image = cv2.cvtColor(cv2.imread(image_file, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        elif self.color_type == 'anydepth':
            image = cv2.imread(image_file, cv2.IMREAD_ANYDEPTH)
        elif self.color_type == 'unchanged':
            image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
        else:
            image = cv2.cvtColor(cv2.imread(image_file, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        assert image is not None, f'empty file: {image_file}'

        if self.to_float:
            image = image.astype(np.float32)

        if self.max_value == 'max':
            image = image / np.max(image)

        if self.force_3chan:
            image = np.stack([image for _ in range(3)], -1)

        results['image'] = image
        results['original_shape'] = (image.shape[1], image.shape[0])
        results['scale_factor'] = 1.0
        return results

    def __call__(self, results):
        image = Image.open(fp=results['image_path'], mode='r', formats=None)
        image = Image.fromarray(obj=image, mode=None)

        if self.to_float:
            image = image.astype(np.float32)

            if self.max_value:
                image /= np.max(image)

        if self.force_3chan:
            image = np.stack([image for _ in range(3)], -1)

        results['image'] = image
        results['original_shape'] = (image.shape[1], image.shape[0])
        results['scale_factor'] = 1.0
        return results


@PIPELINES.register_module
class LoadNumpyImage:
    def __init__(self, to_float=False, max_value=None, force_3chan=False):
        self.to_float = to_float
        self.max_value = max_value
        self.force_3chan = force_3chan

    def __call__(self, results):
        image = np.load(file=results['image_path'])

        if self.to_float:
            image = image.astype(np.float32)

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
    def __init__(self, num_classes=1, from_rel=False, color_type='unchanged'):
        self.num_classes = num_classes
        self.from_rel = from_rel
        if color_type == 'color':
            self.load_option = cv2.IMREAD_COLOR
        elif color_type == 'unchanged':
            self.load_option = cv2.IMREAD_UNCHANGED
        else:
            NotImplementedError('color_type is color or unchanged')

    def __call__(self, results):
        if not ('label_path' in results or 'rle_path' in results):
            return results

        if self.from_rel:
            with open(results['rle_path'], 'r') as f:
                rle = f.read()
            results['label'] = self.rle2mask(mask_rle=rle, shape=results['origin_shape'])
        else:
            label_file = results['label_path']
            results['label'] = cv2.imread(label_file, self.load_option)
        return results
    
    def _to_one_hot(self, label):
        one_hot_label = []
        for i in range(1, self.num_classes + 1):
            one_hot_label.append(label == i)
        
        one_hot_label = np.stack(one_hot_label, axis=2)
        one_hot_label = one_hot_label.astype(np.uint8)
        return one_hot_label

    @staticmethod
    def rle2mask(mask_rle, shape):
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        image = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            image[lo:hi] = 1
        return image.reshape(shape).T

import cv2
from PIL import Image
import numpy as np

from ..builder import PIPELINES

__all__ = [
    'LoadImageFromFile',
    'LoadNumpyImage',
    'LoadAnnotations',
]


@PIPELINES.register_module
class LoadImageFromFile:
    def __init__(self, to_float=True, max_value=None, force_3chan=False, color_type='color', backend='cv2'):
        assert color_type in ('color', 'anydepth', 'unchanged')
        assert backend in ('cv2', 'pil', 'pillow')
        color_types = {'color': cv2.IMREAD_COLOR, 'anydepth': cv2.IMREAD_ANYDEPTH, 'unchanged': cv2.IMREAD_UNCHANGED}
        self.to_float = to_float
        self.color_type = color_types[color_type]
        self.max_value = max_value
        self.force_3chan = force_3chan
        self.backend = backend

    def __call__(self, results):
        if 'image' not in results.keys():
            if self.backend in ('pil', 'pillow'):
                image = Image.open(fp=results['image_path'], mode='r', formats=None)
                results['image'] = Image.fromarray(obj=image, mode=None)

            elif self.backend == 'cv2':
                image = cv2.imread(results['image_path'], self.color_type)
                results['image'] = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)

        if self.to_float:
            results['image'] = results['image'].astype(np.float32)

            if self.max_value == 'max':
                results['image'] /= np.max(results['image'])
            elif isinstance(self.max_value, (float, int)):
                results['image'] /= self.max_value

        if self.force_3chan:
            results['image'] = np.stack([results['image'] for _ in range(3)], -1)

        results['original_shape'] = (results['image'].shape[1], results['image'].shape[0])
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
    def __init__(self, from_rel=False, color_type='unchanged'):
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
import cv2
from PIL import Image
import numpy as np
from ..builder import PIPELINES


@PIPELINES.register_module
class LoadImageFromFile:
    def __init__(self, to_float: bool = True, max_value: float = 255.0, load_option: str = 'color', backend: str = 'cv2', to_3ch: bool = False):
        assert load_option in ('color', 'anydepth', 'unchanged')
        assert backend in ('cv2', 'pil')
        load_options = {'color': cv2.IMREAD_COLOR, 'anydepth': cv2.IMREAD_ANYDEPTH, 'unchanged': cv2.IMREAD_UNCHANGED}
        self.to_float = to_float
        self.max_value = max_value
        self.load_option = load_options[load_option]
        self.backend = backend
        self.to_3ch = to_3ch

    def __call__(self, result: dict):
        if 'image' not in result.keys():
            image = self._load_image(result['image_path'])
            
            if self.to_float:
                image = image.astype(np.float32)
                
                if self.max_value is not None:
                    image /= self.max_value
            
            if self.to_3ch:
                image = np.stack([image for _ in range(3)], axis=2)
            
            result['image'] = image
            result['original_shape'] = (image.shape[1], image.shape[0])
            result['scale_factor'] = 1.0
        else:
            pass
        return result

    def _load_image(self, image_path: str):
        if self.backend == 'cv2':
            image = cv2.imread(filename=image_path, flags=self.load_option)
            image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)
        else:
            image = Image.open(fp=image_path, mode='r', formats=None)
            image = Image.fromarray(obj=image, mode=None)
        return image


@PIPELINES.register_module
class LoadAnnotations:
    def __init__(self, backend: str = 'cv2'):
        assert backend in ('cv2', 'pil')
        self.backend = backend
    
    def __call__(self, result: dict):
        if 'label' not in result.keys():
            label = self._load_mask(result['label_path'])
            if label.ndim == 3:
                label = label.squeeze(axis=2)
            result['label'] = label
        return result
    
    def _load_mask(self, image_path: str):
        if self.backend == 'cv2':
            image = cv2.imread(filename=image_path, flags=cv2.IMREAD_UNCHANGED)
        else:
            image = Image.open(fp=image_path, mode='r', formats=None)
            image = Image.fromarray(obj=image, mode=None)
        return image

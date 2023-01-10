import cv2
import copy
import itertools
import numpy as np
import torch
from .tarnsforms import Resize, Padding, RandomFlipHorizontal, RemovePad, Identity
from ..builder import PIPELINES


@PIPELINES.register_module
class TestTimeAugment:
    """
    TTA pipeline
    >>> tta = TestTimeAugment(
    >>>     input_size=256,
    >>>     input_configs=[
    >>>         {'size': 256, 'flip': False, 'keep_retio': False},
    >>>         {'size': 256, 'flip': True, 'keep_retio': False},
    >>>         {'size': 224, 'flip': False, 'keep_retio': False}
    >>>     ]
    >>> )
    """
    def __init__(
        self,
        input_size: int,
        process_configs: list,
        pad_value: int = 255,
        label_pad_value: int = 255,
    ):
        self.input_size = input_size
        self.process_configs = process_configs
        self.pad_value = pad_value
        self.label_pad_value = label_pad_value
        self._lenght = process_configs.__len__()

        self.sub_processes = list()
        for process_config in process_configs:
            assert isinstance(process_config, dict)
            self.sub_processes.append(SubProcesser(
                input_size=input_size,
                resize_size=process_config['size'],
                flip=process_config['flip'],
                keep_retio=process_config.get('keep_retio', False),
                pad_value=pad_value,
                label_pad_value=label_pad_value
            ))
        
        self.remove_pad = RemovePad()
    
    def __len__(self):
        return self._lenght
        
    def __call__(self, result: dict):
        return self.preprocess(result=result)

    def preprocess(self, result: dict):
        results = list()
        images = list()
        label = result.pop('label')
        for sub_process in self.sub_processes:
            copyed_result = copy.deepcopy(result)
            copyed_result = sub_process(result=copyed_result)
            
            image = np.expand_dims(a=copyed_result.pop('image'), axis=0)
            images.append(image)
            results.append(copyed_result)
        
        images = np.concatenate(images, axis=0).transpose(0, 3, 1, 2)
        images = torch.tensor(images, dtype=torch.float)
        results = {'images': images, 'label': label, 'results': results}
        return results

    def post_process(self, predicts: torch.Tensor, results_list: list):
        """
        Args:
            predicts: Tensor (b * self._lenght, c, w, h)
        """
        predicts = predicts.cpu().detach().float().numpy().transpose(0, 2, 3, 1) # (b * _lenght, w, h, c)
        b = predicts.shape[0]
        batch_size = b // self._lenght
        predicts_list = np.split(predicts, batch_size, axis=0)

        tta_images = list()
        for predicts, results in zip(predicts_list, results_list):
            tta_image = list()
            for predict, result, sub_process in zip(predicts, results, self.sub_processes):
                result['image'] = predict
                result = self.remove_pad(result=result)
                result = sub_process.flip(result=result)

                original_shape = result.pop('original_shape')
                if original_shape != result['image'].shape[:2]:
                    result['image'] = cv2.resize(src=result['image'], dsize=original_shape, interpolation=cv2.INTER_CUBIC)
                tta_image.append(result.pop('image'))
            
            tta_image = np.stack(tta_image, axis=0).mean(axis=0)
            tta_images.append(tta_image)
        return tta_images


class SubProcesser:
    def __init__(
        self,
        input_size: int,
        resize_size: int,
        flip: bool,
        keep_retio: bool,
        pad_value: int,
        label_pad_value: int
    ):
        self.input_size = input_size
        self.resize_size = resize_size
        self.flip = flip
        self.keep_retio = keep_retio

        self.resize = Resize(size=resize_size, keep_retio=keep_retio, ignore_label=True)
        self.padding = Padding(size=input_size, pad_value=pad_value, label_pad_value=label_pad_value)
        self.flip = RandomFlipHorizontal(prob=1.0) if flip else Identity()
    
    def __call__(self, result: dict):
        result = self.resize(result=result)
        result = self.flip(result=result)
        result = self.padding(result=result)
        return result

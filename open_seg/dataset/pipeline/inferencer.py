from copy import deepcopy
import cv2
import numpy as np
import torch
from .transform import Resize, Padding, RemovePad, RandomFlipHorizontal, IdentityTransform
from ..builder import PIPELINES


@PIPELINES.register_module
class SimpleInferencer:
    def __init__(self, input_size, keep_retio=True):
        self.resize = Resize(size=input_size, keep_retio=keep_retio)

    def __call__(self, results):
        return self.pre_process(results=results)

    def pre_process(self, results):
        original_shape = results['original_shape']

        aug_results = self.resize(results=results)

        image = aug_results.pop('image')
        image = image.transpose(2, 0, 1)
        image = torch.tensor(image, dtype=torch.float).unsqueeze(0)

        augmented_results = {'original_shape': original_shape, 'results': aug_results}

        return {'image': image, 'meta': augmented_results}

    def post_process(self, logits, augmented_results):
        logits = logits.sigmoid()
        predict = logits.cpu().detach().squeeze(0).numpy().transpose(1, 2, 0)

        original_shape = augmented_results.pop('original_shape')

        if predict.shape != original_shape:
            predict = cv2.resize(src=predict, dsize=original_shape, interpolation=cv2.INTER_CUBIC)
            if predict.ndim == 2:
                predict = np.expand_dims(a=predict, axis=2)
        return predict


@PIPELINES.register_module
class TestTimeAugment:
    def __init__(self, input_size, scales, flips, keep_retio=True):
        if isinstance(scales, int):
            scales = [scales]

        if isinstance(flips, bool):
            flips = [flips]

        assert scales.__len__() == flips.__len__()
        self.input_size = input_size
        self.scales = scales
        self.flips = flips
        self.keep_retio = keep_retio

        self.padding = Padding(size=input_size, pad_value=0, label_pad_value=0)
        self.remove_pad = RemovePad()
        self.flip_fn = RandomFlipHorizontal(prob=1.0)
        self.resizes = []
        for scale in scales:
            if scale == -1:
                self.resizes.append(IdentityTransform())
            else:
                self.resizes.append(Resize(size=scale, keep_retio=keep_retio))

    def __call__(self, results):
        return self.pre_process(results=results)

    def pre_process(self, results):
        original_shape = results['original_shape']
        augmented_images = []
        augmented_results = {'original_shape': original_shape, 'results': []}

        for resize, flip in zip(self.resizes, self.flips):
            aug_results = deepcopy(results)

            if flip:
                aug_results = self.flip_fn(aug_results)
                aug_results['flip'] = True
            else:
                aug_results['flip'] = False

            aug_results = resize(results=aug_results)
            aug_results = self.padding(results=aug_results)

            image = aug_results.pop('image')
            augmented_images.append(image)
            augmented_results['results'].append(aug_results)

        if augmented_images.__len__() == 1:
            augmented_images = np.expand_dims(a=augmented_images[0], axis=0)
        else:
            augmented_images = np.stack(augmented_images, axis=0)
        augmented_images = augmented_images.transpose((0, 3, 1, 2))
        augmented_images = torch.tensor(augmented_images, dtype=torch.float)
        return {'image': augmented_images, 'meta': augmented_results}

    def post_process(self, logits, augmented_results):
        logits = logits.sigmoid()
        predicts = logits.detach().cpu().float().numpy().transpose(0, 2, 3, 1)
        original_shape = augmented_results.pop('original_shape')
        results = augmented_results.pop('results')
        predict_list = []

        for predict, result in zip(predicts, results):
            result['image'] = predict
            result = self.remove_pad(results=result)
            if result['flip']:
                result = self.flip_fn(result)

            if original_shape != result['image'].shape[:2]:
                result['image'] = cv2.resize(src=result['image'], dsize=original_shape, interpolation=cv2.INTER_CUBIC)
                result['image'] = np.expand_dims(a=result['image'], axis=2)
            predict_list.append(result['image'])

        return sum(predict_list) / predict_list.__len__()

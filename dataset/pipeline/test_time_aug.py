from typing import Union
from copy import deepcopy
import cv2
import torch
from builder import PIPELINES
from dataset.pipeline.transform import Resize, Padding, RemovePad, FlipHorizontal


_UNION = Union[list, tuple]


@PIPELINES.register_module
class TestTimeAugment:
    def __init__(self, scales: _UNION, flips: _UNION, size: int, keep_retio=True, keep_size=True):
        assert len(scales) == len(flips)
        self.scales = scales
        self.flips = flips
        self.keep_size = keep_size

        self.resize = [Resize(size=scale, keep_retio=keep_retio) for scale in scales]
        self.padding = Padding(size=size, pad_value=0, label_pad_value=0)
        self.remove_pad = RemovePad()
        self.flip = FlipHorizontal()

        self.split_size = self.scales.__len__()

    def __call__(self, results):
        return self.pre_process(results=results)

    def pre_process(self, results):
        augmented_images = []
        augmented_results = []
        for scale, flip in zip(self.resize, self.flips):
            aug_results = deepcopy(results)

            if flip:
                aug_results['flip'] = True
                aug_results = self.flip(aug_results)
            else:
                aug_results['flip'] = False

            aug_results = scale(results=aug_results)
            aug_results = self.padding(results=aug_results)

            image = aug_results['image'].transpose(2, 0, 1)
            image = torch.tensor(image, dtype=torch.float)
            augmented_images.append(image)
            augmented_results.append(aug_results)

        if augmented_images.__len__() == 1:
            augmented_images[0].unsqueeze(0)
        else:
            augmented_images = torch.stack(augmented_images, dim=0)
        return {'image': augmented_images, 'meta': augmented_results}

    def post_process(self, logits, augmented_results):
        logits = logits.sigmoid()
        preds = logits.detach().cpu().float().numpy().transpose(0, 2, 3, 1)

        preds_list = []

        origin_shape = augmented_results[0]['original_shape']
        for pred, augmented_result in zip(preds, augmented_results):
            augmented_result['image'] = pred
            augmented_result = self.remove_pad(augmented_result)
            if augmented_result['flip']:
                augmented_result = self.flip(augmented_result)

            image = cv2.resize(src=augmented_result['image'], dsize=origin_shape, interpolation=cv2.INTER_NEAREST)
            preds_list.append(image)

        return sum(preds_list) / preds_list.__len__()


if __name__ == '__main__':
    def main():
        import numpy as np
        import cv2
        import matplotlib.pyplot as plt

        scales = (320, 320, 288, 288)
        flips = (False, True, False, True)
        image = np.zeros((200, 300, 3), dtype=np.uint8) + 1
        image = cv2.putText(
            image,
            text='sample text',
            org=(20, 120),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.0,
            color=(0, 1, 0),
            thickness=2,
            lineType=cv2.LINE_4
        )

        results = {'image': image, 'original_shape': (image.shape[1], image.shape[0])}

        tta = TestTimeAugment(scales=scales, flips=flips, size=320, keep_retio=True, keep_size=True)

        augmented_images, augmented_results = tta.pre_process(results=results)

        # for augmented_image, augmented_result in zip(augmented_images, augmented_results):
        #     print(f'shape: {augmented_image.shape} - dtype: {augmented_image.dtype} - ID: {id(augmented_result)}')
        #     print('flip: ', augmented_result['flip'])
        #     augmented_image = augmented_image.numpy().transpose(1, 2, 0)
        #     plt.imshow(augmented_image)
        #     plt.show()

        predict = tta.post_process(logits=augmented_images, augmented_results=augmented_results)

        plt.imshow(predict)
        plt.show()

    main()

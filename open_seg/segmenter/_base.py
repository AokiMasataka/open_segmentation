from copy import deepcopy
from abc import ABCMeta, abstractmethod

from torch.nn import functional
import torch
from open_seg.utils import conv3x3, conv1x1, init_weight


class SegmenterBase(torch.nn.Module, metaclass=ABCMeta):
    def __init__(self, num_classes, test_config=None, init_config=None):
        super().__init__()
        self._is_init = False
        self.init_config = deepcopy(init_config)
        self.test_config = test_config
        self.num_classes = num_classes

        if self.init_config is not None:
            self.load_weight()
        else:
            self.init_weight()

    @abstractmethod
    def forward_train(self, image, label):
        pass

    @abstractmethod
    def forward_test(self, image, label):
        pass

    @property
    def is_init(self):
        return self._is_init

    def init_weight(self):
        self._is_init = True

    def load_weight(self):
        weight_path = self.init_config.get('weight_path', None)
        if weight_path is not None:
            state_dict = torch.load(weight_path, map_location='cpu')
            miss_match_key = self.load_state_dict(state_dict, strict=False)
            print(miss_match_key)

        self._is_init = True

    @torch.inference_mode()
    def slide_inference(self, image):
        h_stride, w_stride = self.test_config['stride']
        h_crop, w_crop = self.test_config['crop_size']
        num_classes = self.num_classes

        batch, _, h_img, w_img = image.size()
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = image.new_zeros((batch, num_classes, h_img, w_img))
        count_mat = image.new_zeros((batch, 1, h_img, w_img))
        crop_image_list = []
        pad_meta_list = []

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                count_mat[:, :, y1:y2, x1:x2] += 1
                crop_image_list.append(image[:, :, y1:y2, x1:x2])
                pad_meta_list.append((int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2)))

        crop_image = torch.cat(crop_image_list, dim=0)
        crop_logit_list = [image for image in self(crop_image)]

        for crop_logit, pad_meta in zip(crop_logit_list, pad_meta_list):
            preds += functional.pad(input=crop_logit, pad=pad_meta, mode='constant', value=0)

        assert (count_mat == 0).sum() == 0
        preds = preds / count_mat
        return preds


class SegmentationHead(torch.nn.Module):
    def __init__(self, in_channels, num_classes, hidden_rate=1.0):
        super(SegmentationHead, self).__init__()
        hidden_dim = int(in_channels * hidden_rate)
        self.conv1 = conv3x3(in_channels, hidden_dim).apply(init_weight)
        self.act = torch.nn.ELU(True)
        self.conv2 = conv1x1(hidden_dim, num_classes).apply(init_weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        return self.conv2(x)

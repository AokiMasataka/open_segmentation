from copy import deepcopy
from typing import Optional
from abc import ABCMeta, abstractmethod

import torch
from open_seg.utils import conv3x3, conv1x1, init_weight


class SegmenterBase(torch.nn.Module, metaclass=ABCMeta):
    def __init__(self, init_config: Optional[dict] = None):
        super().__init__()
        self._is_init = False
        self.init_config = deepcopy(init_config)

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

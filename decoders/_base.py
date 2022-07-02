from copy import deepcopy
from typing import Optional
from abc import ABCMeta, abstractmethod

import torch
from torch import nn
from torch.nn import functional


__all__ = ['DecoderBase', 'DecoderBasicBlock', 'DecoderBottleneckBlock']


class DecoderBase(torch.nn.Module, metaclass=ABCMeta):
    def __init__(self, init_config: Optional[dict] = None):
        super().__init__()
        self._is_init = False
        self.init_config = deepcopy(init_config)

        if self.init_config is not None:
            self.load_weight()
        else:
            self.init_weight()

    @abstractmethod
    def decoder_out_dim(self):
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


class DecoderBasicBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, scale_factor=2, mode='nearest'):
        super(DecoderBasicBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x, skip=None):
        x = functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        if skip is not None:
            x = torch.cat(tensors=(x, skip), dim=1)
        x = self.block(x)
        return x


class DecoderBottleneckBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, scale_factor=2, mode='nearest'):
        super(DecoderBottleneckBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels // 2, kernel_size=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x, skip=None):
        x = functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        if skip is not None:
            x = torch.cat(tensors=(x, skip), dim=1)
        x = self.block(x) + x
        x = self.relu(x)
        return x

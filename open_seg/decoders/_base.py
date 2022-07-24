from copy import deepcopy
from typing import Optional
from abc import ABCMeta, abstractmethod

import torch
from torch import nn
from torch.nn import functional

from open_seg.utils import conv3x3, conv1x1, init_weight


__all__ = ['DecoderBase', 'DecoderBasicBlock', 'DecoderBottleneckBlock', 'DECODER_BLOCK', 'ASPP']


class DecoderBase(torch.nn.Module, metaclass=ABCMeta):
    def __init__(self, init_config: Optional[dict] = None):
        super().__init__()
        self._is_init = False
        self.init_config = deepcopy(init_config)
        self.deep_supervision = False

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
    def __init__(self, in_channels, out_channels, scale_factor=2, mode='nearest'):
        super(DecoderBasicBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(in_channel=in_channels, out_channel=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            conv3x3(in_channel=out_channels, out_channel=out_channels),
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
    def __init__(self, in_channels, out_channels, scale_factor=2, mode='nearest'):
        super(DecoderBottleneckBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=(1, 1), padding=(0, 0), bias=False),
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


class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channel, reduction):
        super().__init__()
        self.global_maxpool = nn.AdaptiveMaxPool2d(1)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            conv1x1(in_channel=in_channel, out_channel=in_channel // reduction).apply(init_weight),
            nn.ReLU(inplace=True),
            conv1x1(in_channel=in_channel // reduction, out_channel=in_channel).apply(init_weight)
        )

    def forward(self, inputs):
        x1 = self.global_maxpool(inputs)
        x2 = self.global_avgpool(inputs)
        x1 = self.fc(x1)
        x2 = self.fc(x2)
        x = torch.sigmoid(x1 + x2)
        return x


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3x3 = conv3x3(in_channel=2, out_channel=1).apply(init_weight)

    def forward(self, inputs):
        x1, _ = torch.max(inputs, dim=1, keepdim=True)
        x2 = torch.mean(inputs, dim=1, keepdim=True)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv3x3(x)
        x = torch.sigmoid(x)
        return x


class CBAM(nn.Module):
    def __init__(self, in_channel, reduction):
        super().__init__()
        self.channel_attention = ChannelAttentionModule(in_channel=in_channel, reduction=reduction)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, inputs):
        x = inputs * self.channel_attention(inputs=inputs)
        x = x * self.spatial_attention(inputs=x)
        return x


class DecoderBasicCbamBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, mode='nearest'):
        super().__init__()
        self.block = nn.Sequential(
            conv3x3(in_channel=in_channels, out_channel=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            conv3x3(in_channel=out_channels, out_channel=out_channels),
            CBAM(in_channel=out_channels, reduction=16),
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


# TODO ASPP Module
class ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, groups=1):
        super(ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(
            in_channels=inplanes,
            out_channels=planes,
            kernel_size=(kernel_size, kernel_size),
            stride=(1, 1),
            padding=padding,
            dilation=dilation,
            bias=False,
            groups=groups
        )
        self.bn = nn.BatchNorm2d(planes)
        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        return functional.relu(x, inplace=True)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, in_channels=512, mid_channels=256, out_channels=None, dilations=(6, 12, 18, 24)):
        super(ASPP, self).__init__()
        self.aspps = [ASPPModule(in_channels, mid_channels, 1, padding=0, dilation=1)] + \
                     [ASPPModule(in_channels, mid_channels, 3, padding=d, dilation=d, groups=4) for d in dilations]
        self.aspps = nn.ModuleList(self.aspps)
        self.global_pool = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        out_channels = out_channels if out_channels is not None else mid_channels
        in_channels = mid_channels * (2 + len(dilations))
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_ch, kernel_size=(1, 1), bias=False)
        self._init_weight()

    def forward(self, x):
        x0 = self.global_pool(x)
        xs = [aspp(x) for aspp in self.aspps]
        x0 = functional.interpolate(x0, size=xs[0].size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x0] + xs, dim=1)
        return self.out_conv(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


DECODER_BLOCK = {
    'basic': DecoderBasicBlock,
    'basic_cbm': DecoderBasicCbamBlock,
    'bottleneck': DecoderBottleneckBlock,
}

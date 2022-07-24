# https://github.com/tikutikutiku/kaggle-hubmap/blob/main/src/models.py
import torch
from torch import nn
from torch.nn import functional

from ._base import DecoderBase, DECODER_BLOCK, conv3x3
from open_seg.builder import DECODERS


@DECODERS.register_module
class UnetHypercolum(DecoderBase):
    def __init__(self, encoder_channels, decoder_channels, n_blocks=5, block_type='basic_cbm'):
        super(UnetHypercolum, self).__init__()
        assert n_blocks == len(decoder_channels)
        assert block_type in DECODER_BLOCK.keys()

        self._encoder_channels = encoder_channels
        self._decoder_channels = decoder_channels

        encoder_channels = list(encoder_channels)
        decoder_channels = list(decoder_channels)
        encoder_channels.reverse()

        head_channels = encoder_channels[0]
        in_channels = [head_channels] + decoder_channels[:-1]
        skip_channels = encoder_channels[1:] + [0]
        out_channels = decoder_channels
        in_channels = [in_ch + skip_ch for in_ch, skip_ch in zip(in_channels, skip_channels)]

        block = DECODER_BLOCK[block_type]
        blocks = [block(in_ch, out_ch) for in_ch, out_ch in zip(in_channels, out_channels)]
        self.blocks = nn.ModuleList(blocks)
        self.scales = [2 ** i for i in range(1, n_blocks)][::-1]
        self.fpn = nn.ModuleList([
            nn.Sequential(
                conv3x3(in_channel=in_ch, out_channel=out_ch * 2),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_ch * 2),
                conv3x3(in_channel=out_ch * 2, out_channel=out_ch)
            ) for in_ch, out_ch in zip(decoder_channels[:-1], [16] * (decoder_channels.__len__() - 1))
        ])
        self._decoder_out_dim = sum([16] * (decoder_channels.__len__() - 1) + [decoder_channels[-1]])

    def forward(self, features):
        x = features.pop()
        features.reverse()

        hypercolums = []
        for feature, decoder_block, scale, fpn in zip(features, self.blocks, self.scales, self.fpn):
            x = decoder_block(x=x, skip=feature)
            hypercolums.append(fpn(functional.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)))

        hypercolums.append(self.blocks[-1](x=x))
        return torch.cat(hypercolums, dim=1)

    def decoder_out_dim(self):
        return self._decoder_out_dim


@DECODERS.register_module
class UnetHypercolumDeepSupervision(DecoderBase):
    def __init__(self, encoder_channels, decoder_channels, n_blocks=5, block_type='basic_cbm'):
        super(UnetHypercolumDeepSupervision, self).__init__()
        assert n_blocks == len(decoder_channels)
        assert block_type in DECODER_BLOCK.keys()

        self._encoder_channels = encoder_channels
        self._decoder_channels = decoder_channels

        encoder_channels = list(encoder_channels)
        decoder_channels = list(decoder_channels)
        encoder_channels.reverse()

        head_channels = encoder_channels[0]
        in_channels = [head_channels] + decoder_channels[:-1]
        skip_channels = encoder_channels[1:] + [0]
        out_channels = decoder_channels
        in_channels = [in_ch + skip_ch for in_ch, skip_ch in zip(in_channels, skip_channels)]

        block = DECODER_BLOCK[block_type]
        blocks = [block(in_ch, out_ch) for in_ch, out_ch in zip(in_channels, out_channels)]
        self.blocks = nn.ModuleList(blocks)
        self.deep_visions = [conv3x3(in_channel=out_ch, out_channel=1) for out_ch in out_channels]
        self.scales = [2 ** i for i in range(1, n_blocks)][::-1]

    def forward(self, features):
        x = features.pop()
        features.reverse()

        hypercolums = []
        deep_visions = []
        for feature, decoder_block, deep_vision, scale in zip(features, self.blocks, self.deep_visions, self.scales):
            x = decoder_block(x=x, skip=feature)
            hypercolums.append(functional.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False))
            deep_visions.append(deep_vision(hypercolums[-1]))

        hypercolums.append(self.blocks[-1](x=x))
        return torch.cat(hypercolums, dim=1), deep_visions

    def inference(self, features):
        x = features.pop()
        features.reverse()

        hypercolums = []
        for feature, decoder_block, scale in zip(features, self.blocks, self.scales):
            x = decoder_block(x=x, skip=feature)
            hypercolums.append(functional.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False))

        hypercolums.append(self.blocks[-1](x=x))
        return torch.cat(hypercolums, dim=1)

    def decoder_out_dim(self):
        return self._decoder_channels[-1]

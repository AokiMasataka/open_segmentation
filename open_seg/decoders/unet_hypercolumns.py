# https://github.com/tikutikutiku/kaggle-hubmap/blob/main/src/models.py
import torch
from torch import nn
from torch.nn import functional

from ._base import DecoderBase, DECODER_BLOCK
from open_seg.builder import DECODERS


@DECODERS.register_module
class UnetHypercolum(DecoderBase):
    def __init__(self, encoder_channels, decoder_channels, n_blocks=5, block_type='cbam'):
        super(UnetHypercolum, self).__init__()
        assert n_blocks == len(decoder_channels) - 1
        assert block_type in DECODER_BLOCK.keys()

        self._encoder_channels = encoder_channels
        self._decoder_channels = decoder_channels

        encoder_channels = list(encoder_channels)
        decoder_channels = list(decoder_channels)
        encoder_channels.reverse()

        decoder_in_channels = [
            dec_ch + enc_ch for dec_ch, enc_ch in zip(decoder_channels[:-1], encoder_channels[:4] + [0])
        ]

        block = DECODER_BLOCK[block_type]
        blocks = [
            block(dec_in, dec_out) for dec_in, dec_out in zip(decoder_in_channels, decoder_channels[1:])
        ]

        self.blocks = nn.ModuleList(blocks)
        self.center_block = nn.Conv2d(
            in_channels=encoder_channels[0],
            out_channels=decoder_channels[0],
            kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )

        self.scales = (2, 4, 8, 16)
        self._decoder_out_dim = sum(decoder_channels[1:])

    def forward(self, features):
        features = features[1:]
        features.reverse()

        decoder_features = [self.center_block(features[0])]

        for block, feature in zip(self.blocks, features):
            decoder_features.append(block(torch.cat([feature, decoder_features[-1]], dim=1)))

        decoder_features.append(self.blocks[-1](decoder_features[-1]))
        decoder_features = decoder_features[1:]

        for index, scale in zip(range(decoder_features.__len__() - 1), self.scales[::-1]):
            decoder_features[index] = functional.interpolate(decoder_features[index], scale_factor=scale, mode='bilinear', align_corners=True)

        hypercol = torch.cat(decoder_features, dim=1)
        # logits = self.final_conv(hypercol)
        return hypercol

    def decoder_out_dim(self):
        return self._decoder_out_dim

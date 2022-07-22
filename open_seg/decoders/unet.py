from torch import nn
from torch.nn import functional

from ._base import DecoderBase, DECODER_BLOCK, conv1x1
from open_seg.builder import DECODERS


@DECODERS.register_module
class Unet(DecoderBase):
    def __init__(self, encoder_channels, decoder_channels, n_blocks=5, block_type='basic'):
        super(Unet, self).__init__()
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

    def forward(self, features):
        x = features.pop()
        features.reverse()

        for decoder_block, feature in zip(self.blocks, features):
            x = decoder_block(x=x, skip=feature)

        x = self.blocks[-1](x=x)
        return x

    def decoder_out_dim(self):
        return self._decoder_channels[-1]


@DECODERS.register_module
class UnetDeepVision(DecoderBase):
    def __init__(self, encoder_channels, decoder_channels, n_blocks=5, block_type='basic'):
        super(UnetDeepVision, self).__init__()
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

        self.deep_supervision_blocks = nn.ModuleList([conv1x1(in_ch, 1) for in_ch in out_channels[::-1][1:]])

    def forward(self, features):
        x = features.pop()
        features.reverse()

        deepsupervision = []
        for decoder_block, feature in zip(self.blocks, features):
            x = decoder_block(x=x, skip=feature)
            deepsupervision.append(x)

        x = self.blocks[-1](x=x)
        deepsupervision.reverse()
        for index, deep in enumerate(deepsupervision):
            scale = 2 ** (index + 1)
            deep = functional.interpolate(deep, scale_factor=scale, mode='bilinear', align_corners=False)
            deepsupervision[index] = self.deep_supervision_blocks[index](deep)
        return x, deepsupervision

    def decoder_out_dim(self):
        return self._decoder_channels[-1]
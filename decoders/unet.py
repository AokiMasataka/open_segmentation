import torch
from torch import nn
from torch.nn import functional
from decoders.base import DecoderBasicBlock, DecoderBottleneckBlock

from builder import DECODERS



@DECODERS.register_module
class Unet(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, n_blocks=5, block_type='basic'):
        super(Unet, self).__init__()
        assert n_blocks == len(decoder_channels)

        self._encoder_channels = encoder_channels
        self._decoder_channels = decoder_channels

        # remove first skip with same spatial resolution
        # encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        # combine decoders keyword arguments
        block = DecoderBasicBlock
        if block_type == 'basic':
            block = DecoderBasicBlock
        elif block_type == 'bottleneck':
            block = DecoderBottleneckBlock
        else:
            ValueError(f'block_type is basic or bottleneck')

        blocks = [
            block(in_ch, skip_ch, out_ch) for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):
        features = features[::-1]  # reverse channels to start from head of encoder

        x = features[0]
        skips = features[1:]

        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
        return x

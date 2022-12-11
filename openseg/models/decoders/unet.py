from torch import nn
from .base import DecoderBase, ASPP, CenterBlock, DecoderBlock
from ..builder import DECODERS


@DECODERS.register_module
class Unet(DecoderBase):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            scale_factors=None,
            n_blocks=5,
            attention_type='scse',
            center_block_type='basic',
            inter_mode='nearest'
    ):
        super(Unet, self).__init__()
        assert n_blocks == len(decoder_channels)
        assert n_blocks == len(encoder_channels)

        self._encoder_channels = encoder_channels
        self._decoder_channels = decoder_channels

        encoder_channels = list(encoder_channels)
        decoder_channels = list(decoder_channels)

        if scale_factors is None:
            scale_factors = (4,) + tuple(2 for _ in range(n_blocks - 1))
        scale_factors = list(scale_factors)
        encoder_channels.reverse()
        scale_factors.reverse()

        # computing blocks input and output channels
        in_channels = [encoder_channels[0]] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        in_channels = [in_ch + skip_ch for in_ch, skip_ch in zip(in_channels, skip_channels)]
        out_channels = decoder_channels

        blocks = [DecoderBlock(in_ch, out_ch, scale_factor, mode=inter_mode, attention_type=attention_type)
                  for in_ch, out_ch, scale_factor in zip(in_channels, out_channels, scale_factors)]
        self.blocks = nn.ModuleList(blocks)

        if center_block_type == 'aspp':
            self.center = ASPP(in_channels=encoder_channels[0], out_channels=encoder_channels[0])
        elif center_block_type == 'basic':
            self.center = CenterBlock(in_channels=encoder_channels[0], out_channels=encoder_channels[0])

    def forward(self, features):
        x = self.center(features.pop())
        features.reverse()

        for decoder_block, feature in zip(self.blocks, features):
            x = decoder_block(x=x, skip=feature)

        x = self.blocks[-1](x=x)
        return x

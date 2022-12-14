import torch
from torch import nn
from .blocks import ResBlock, CenterBlock, TransformerCenterBlock
from .base import DecoderBase
from ..builder import DECODERS


@DECODERS.register_module
class UnetHead(DecoderBase):
    def __init__(
        self,
        encoder_dims,
        decoder_dims,
        num_blocks: int,
        center_block_config: dict = None,
        layers: int = 1,
        eps: float = 1e-6,
    ):
        super(UnetHead, self).__init__()
        assert num_blocks == encoder_dims.__len__() == decoder_dims.__len__()
        
        encoder_dims = list(encoder_dims)
        decoder_dims = list(decoder_dims)
        encoder_dims.reverse()

        blocks = list()
        prov_dim = encoder_dims[0]
        for encoder_dim, decoder_dim, i in zip(encoder_dims, decoder_dims, range(num_blocks)):
            upsample = 4 if i + 1 == num_blocks else 2
            blocks.append(
                ResBlock(
                    in_dim=prov_dim + encoder_dim,
                    out_dim=decoder_dim,
                    layers=layers,
                    eps=eps,
                    activation='gelu',
                    upsample=upsample
                 )
            )
            prov_dim = decoder_dim
        
        self.blocks = nn.ModuleList(blocks)
        self.center_block = build_center_block(center_block_config)
    
    def forward(self, features):
        features.reverse()
        
        x = self.center_block(features[0])
        for block, feature in zip(self.blocks, features):
            x = torch.cat((x, feature), dim=1)
            x = block(x)
        return x


def build_center_block(center_block_config: dict):
    if center_block_config is None:
        return nn.Identity()
    center_blocks = {'CenterBlock': CenterBlock, 'TransformerCenterBlock': TransformerCenterBlock}
    center_block = center_blocks[center_block_config.pop('type')]
    return center_block(**center_block_config)
    
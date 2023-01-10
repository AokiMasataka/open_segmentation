import torch
from torch import nn
from .base import DecoderBase
from .blocks import ResBlock, CenterBlock, TransformerCenterBlock, UpsampleBlock
from ..builder import DECODERS


@DECODERS.register_module
class UnetHead(DecoderBase):
    """
    >>> import torch
    >>> num_classes = 10
    >>> encoder_output_dims = (64, 128, 256, 512)
    >>> head = UnetHead(
    >>>     num_classes=num_classes,
    >>>     encoder_dims=encoder_output_dims,
    >>>     decoder_dims=(32, 64, 128, 256),
    >>>     layers=1,
    >>>     init_config=None
    >>> )
    >>> features = [torch.rand(1, dim, s, s) for dim, s in zip(encoder_output_dims, (64, 32, 16, 8))]
    >>> output = head(features=features)
    >>> print(output.shape)
    >>> (1, 10, 256, 256)
    """
    def __init__(
        self,
        num_classes: int,
        encoder_dims: tuple = (64, 128, 256, 512),
        decoder_dims: tuple = (32, 64, 128, 256),
        center_block_config: dict = dict(type='CenterBlock', input_dim=512),
        layers: int = 1,
        init_config: dict = None,
        eps: float = 1e-5,
    ):
        super(UnetHead, self).__init__(init_config=init_config)
        assert encoder_dims.__len__() == decoder_dims.__len__()
        self._num_classes = num_classes
        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims

        encoder_dims = tuple(reversed(encoder_dims))
        decoder_dims = tuple(reversed(decoder_dims))

        self.blocks = nn.ModuleList()
        prov_dim = encoder_dims[0]
        for input_dim, output_dim, index in zip(encoder_dims, decoder_dims, range(encoder_dims.__len__())):
            upsample = 4 if index + 1 == encoder_dims.__len__() else 2
            self.blocks.append(
                UpsampleBlock(
                    input_dim=input_dim + prov_dim,
                    output_dim=output_dim,
                    num_layers=layers,
                    scale=upsample,
                    mode='nearest',
                    dropout=0.1,
                    eps=eps
                )
            )
            prov_dim = output_dim
        
        self.center_block = build_center_block(center_block_config)
        self.head = nn.Sequential(
            nn.Conv2d(decoder_dims[-1], decoder_dims[-1] * 2, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.1),
            nn.BatchNorm2d(num_features=decoder_dims[-1] * 2, eps=eps),
            nn.Conv2d(decoder_dims[-1] * 2, num_classes, kernel_size=(1, 1), stride=(1, 1), padding=0),
        )

    
    def forward(self, features: list):
        features.reverse()
        x = self.center_block(features[0])
        for block, feature in zip(self.blocks, features):
            x = torch.cat((x, feature), dim=1)
            x = block(x)

        x = self.head(x)
        return x


def build_center_block(center_block_config: dict):
    if center_block_config is None:
        return nn.Identity()
    center_blocks = {'CenterBlock': CenterBlock, 'TransformerCenterBlock': TransformerCenterBlock}
    center_block = center_blocks[center_block_config.pop('type')]
    return center_block(**center_block_config)
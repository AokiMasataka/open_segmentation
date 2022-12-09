import torch
from torch import nn
from torch.nn import functional
from .base import DecoderBase, DecoderBlock
from ..builder import DECODERS


class MLP(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        return self.proj(x.flatten(2).transpose(1, 2))


@DECODERS.register_module
class SegFormerHead(DecoderBase):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, in_channels=(32, 64, 160, 256), embedding_dim=256, drop_prob=0.0):
        super(SegFormerHead, self).__init__()
        self._decoder_channels = [embedding_dim]
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_channels=embedding_dim * 4, out_channels=embedding_dim, kernel_size=(1, 1)),
            nn.BatchNorm2d(num_features=embedding_dim),
        )

        self.dropout = nn.Dropout2d(p=drop_prob)
        self.linear_pred = DecoderBlock(in_ch=embedding_dim, out_ch=embedding_dim, scale_factor=4, mode='bilinear')

    def forward(self, x):
        c1, c2, c3, c4 = x
        interpolate_size = c1.size()[2:]

        n, _, _, _ = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = functional.interpolate(input=_c4, size=interpolate_size, mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = functional.interpolate(input=_c3, size=interpolate_size, mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = functional.interpolate(input=_c2, size=interpolate_size, mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.linear_pred(self.dropout(_c))
        return x

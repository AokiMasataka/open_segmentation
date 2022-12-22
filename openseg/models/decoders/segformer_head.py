import torch
from torch import nn
from torch.nn import functional
from .base import DecoderBase
from .blocks import ResBlock
from ..builder import DECODERS


class MLP(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=768):
        super(MLP, self).__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        return self.proj(x.flatten(2).transpose(1, 2))


@DECODERS.register_module
class SegFormerHead(DecoderBase):
    def __init__(
        self,
        num_classes: int,
        input_dims: tuple = (32, 64, 160, 256),
        embedding_dim: int = 256,
        drop_prob: float = 0.0,
        eps: float = 1e-6
    ):
        super(SegFormerHead, self).__init__()
        c1_in_dim, c2_in_dim, c3_in_dim, c4_in_dim = input_dims

        self.linear_c1 = MLP(input_dim=c1_in_dim, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_dim, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_dim, embed_dim=embedding_dim)
        self.linear_c4 = MLP(input_dim=c4_in_dim, embed_dim=embedding_dim)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_channels=embedding_dim * 4, out_channels=embedding_dim, kernel_size=(1, 1)),
            nn.BatchNorm2d(num_features=embedding_dim, eps=eps),
        )

        self.dropout = nn.Dropout2d(p=drop_prob)
        self.linear_pred = ResBlock(in_dim=embedding_dim, out_dim=embedding_dim, layers=1, upsample=4, activation='gelu', extend=2)
        self.head = nn.Conv2d(embedding_dim, num_classes, kernel_size=(1, 1), stride=(1, 1), padding=0)

    def forward(self, x):
        c1, c2, c3, c4 = x
        n, _, _, _ = c4.shape
        interpolate_size = c1.size()[2:]

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = functional.interpolate(input=_c4, size=interpolate_size, mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = functional.interpolate(input=_c3, size=interpolate_size, mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = functional.interpolate(input=_c2, size=interpolate_size, mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.linear_pred(self.dropout(_c))
        x = self.head(x)
        return x

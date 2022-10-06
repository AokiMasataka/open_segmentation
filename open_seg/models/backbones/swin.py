import timm
from torch import nn
from ..builder import BACKBONES


class SwinTransformer(nn.Module):
    def __init__(self, base_net, n_blocks=4):
        super(SwinTransformer, self).__init__()
        self.n_blocks = n_blocks
        self.patch_embed = base_net.patch_embed
        self.pos_drop = base_net.pos_drop
        self.layers = base_net.layers
        self.scale_factors = (4,) + tuple(2 for _ in range(n_blocks - 1))

    def forward(self, x):
        features = [None]
        x = self.patch_embed(x)
        features.append(x)
        x = self.pos_drop(x)

        for index, block in zip(range(self.n_blocks - 1), self.layers):
            x = block(x)
            features.append(x)

        return features


@BACKBONES.register_module
def swin_tiny(pretrained=False, n_blocks=4):
    base_net = timm.create_model('swin_tiny_patch4_window7_224', pretrained=pretrained)
    return SwinTransformer(base_net=base_net, n_blocks=n_blocks)


@BACKBONES.register_module
def swin_small(pretrained=False, n_blocks=4):
    base_net = timm.create_model('swin_small_patch4_window7_224', pretrained=pretrained)
    return SwinTransformer(base_net=base_net, n_blocks=n_blocks)


@BACKBONES.register_module
def swin_base(pretrained=False, n_blocks=4):
    base_net = timm.create_model('swin_base_patch4_window12_384', pretrained=pretrained)
    return SwinTransformer(base_net=base_net, n_blocks=n_blocks)


@BACKBONES.register_module
def swin_large(pretrained=False, n_blocks=4):
    base_net = timm.create_model('swin_large_patch4_window12_384', pretrained=pretrained)
    return SwinTransformer(base_net=base_net, n_blocks=n_blocks)
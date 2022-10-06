from torch import nn
import timm
from ..builder import BACKBONES


class ConvNeXt(nn.Module):
    def __init__(self, base_net, n_blocks=5):
        super(ConvNeXt, self).__init__()
        assert n_blocks <= 4
        base_net.stages[0].downsample = base_net.stem
        blocks = [*base_net.stages]

        self.blocks = nn.ModuleDict({
            f'block{index}': block for index, block in zip(range(n_blocks), blocks)
        })

        _out_channels = tuple(block.blocks[0].conv_dw.in_channels for block in blocks)
        self._out_channels = (0,) + _out_channels
        self.scale_factors = (4,) + tuple(2 for _ in range(n_blocks - 1))

    def forward(self, x):
        features = list()
        for block in self.blocks.values():
            x = block(x)
            features.append(x)
        return features

    def out_channels(self):
        return self._out_channels


@BACKBONES.register_module
def convnext_tiny(pretrained=False, drop_path_rate=0.0, n_blocks=4):
    base_net = timm.create_model(
        model_name='convnext_tiny',
        pretrained=pretrained,
        drop_path_rate=drop_path_rate,
    )
    return ConvNeXt(base_net=base_net, n_blocks=n_blocks)


@BACKBONES.register_module
def convnext_small(pretrained=False, drop_path_rate=0.0, n_blocks=4):
    base_net = timm.create_model(
        model_name='convnext_small',
        pretrained=pretrained,
        drop_path_rate=drop_path_rate,
    )
    return ConvNeXt(base_net=base_net, n_blocks=n_blocks)


@BACKBONES.register_module
def convnext_base(pretrained=False, drop_path_rate=0.0, n_blocks=4):
    base_net = timm.create_model(
        model_name='convnext_base',
        pretrained=pretrained,
        drop_path_rate=drop_path_rate,
    )
    return ConvNeXt(base_net=base_net, n_blocks=n_blocks)


@BACKBONES.register_module
def convnext_large(pretrained=False, drop_path_rate=0.0, n_blocks=4):
    base_net = timm.create_model(
        model_name='convnext_large',
        pretrained=pretrained,
        drop_path_rate=drop_path_rate,
    )
    return ConvNeXt(base_net=base_net, n_blocks=n_blocks)
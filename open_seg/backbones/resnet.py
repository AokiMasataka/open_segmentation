from torch import nn
from torchvision.models import resnet
from open_seg.backbones._base import BackboneBase
from open_seg.builder import BACKBONES


class ResNet(BackboneBase):
    def __init__(self, model, n_blocks=5, init_config=None):
        super(ResNet, self).__init__(init_config=init_config)

        blocks = [
            nn.Sequential(model.conv1, model.bn1, model.relu),
            nn.Sequential(model.maxpool, *model.layer1),
            model.layer2,
            model.layer3,
            model.layer4,
        ]

        self.blocks = nn.ModuleDict({
            f'block{index}': block for index, block in zip(range(n_blocks), blocks)
        })

        block1_channel = blocks[0][0].out_channels
        self.channels = tuple([3, block1_channel] + [block[-1].conv1.in_channels for block in blocks[1:n_blocks]])

    def forward(self, x):
        features = [x]
        for block in self.blocks.values():
            x = block(x)
            features.append(x)
        return features

    def out_channels(self):
        return self._out_channels


@BACKBONES.register_module
def resnet18(pretrained=None, n_blocks=5, init_config=None):
    model = resnet.resnet18(weights=pretrained)
    return ResNet(model=model, n_blocks=n_blocks, init_config=init_config)


@BACKBONES.register_module
def resnet34(pretrained=None, n_blocks=5, init_config=None):
    model = resnet.resnet34(weights=pretrained)
    return ResNet(model=model, n_blocks=n_blocks, init_config=init_config)


@BACKBONES.register_module
def resnet50(pretrained=None, n_blocks=5, init_config=None):
    model = resnet.resnet50(weights=pretrained)
    return ResNet(model=model, n_blocks=n_blocks, init_config=init_config)


@BACKBONES.register_module
def resnet101(pretrained=None, n_blocks=5, init_config=None):
    model = resnet.resnet101(weights=pretrained)
    return ResNet(model=model, n_blocks=n_blocks, init_config=init_config)


@BACKBONES.register_module
def resnet152(pretrained=None, n_blocks=5, init_config=None):
    model = resnet.resnet152(weights=pretrained)
    return ResNet(model=model, n_blocks=n_blocks, init_config=init_config)


@BACKBONES.register_module
def resnext50_32x4d(pretrained=None, n_blocks=5, init_config=None):
    model = resnet.resnext50_32x4d(weights=pretrained)
    return ResNet(model=model, n_blocks=n_blocks, init_config=init_config)


@BACKBONES.register_module
def resnext101_32x8d(pretrained=None, n_blocks=5, init_config=None):
    model = resnet.resnext101_32x8d(weights=pretrained)
    return ResNet(model=model, n_blocks=n_blocks, init_config=init_config)

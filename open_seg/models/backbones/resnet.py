import torch
from torch import nn
from torchvision.models import resnet
from ..builder import BACKBONES


class ResNet(nn.Module):
    def __init__(self, model, n_blocks=5):
        super(ResNet, self).__init__()

        blocks = [
            nn.Sequential(model.conv1, model.bn1, nn.ReLU(inplace=True)),
            nn.Sequential(model.maxpool, *model.layer1),
            model.layer2,
            model.layer3,
            model.layer4,
        ]

        self.blocks = nn.ModuleDict({
            f'block{index}': block for index, block in zip(range(n_blocks), blocks)
        })

        block1_channel = blocks[0][0].out_channels
        self._out_channels = tuple([block1_channel] + [block[-1].conv1.in_channels for block in blocks[1:n_blocks]])
        self.scale_factors = (2 for _ in range(n_blocks))

    def forward(self, x):
        features = []
        for block in self.blocks.values():
            x = block(x)
            features.append(x)
        return features

    def out_channels(self):
        return self._out_channels


@BACKBONES.register_module
def resnet18(pretrained=None, n_blocks=5):
    model = resnet.resnet18(weights=pretrained)
    return ResNet(model=model, n_blocks=n_blocks)


@BACKBONES.register_module
def resnet34(pretrained=None, n_blocks=5):
    model = resnet.resnet34(weights=pretrained)
    return ResNet(model=model, n_blocks=n_blocks)


@BACKBONES.register_module
def resnet50(pretrained=None, n_blocks=5):
    model = resnet.resnet50(weights=pretrained)
    return ResNet(model=model, n_blocks=n_blocks)


@BACKBONES.register_module
def resnet101(pretrained=None, n_blocks=5):
    model = resnet.resnet101(weights=pretrained)
    return ResNet(model=model, n_blocks=n_blocks)


@BACKBONES.register_module
def resnet152(pretrained=None, n_blocks=5):
    model = resnet.resnet152(weights=pretrained)
    return ResNet(model=model, n_blocks=n_blocks)


@BACKBONES.register_module
def resnext50_32x4d(pretrained=None, n_blocks=5):
    model = resnet.resnext50_32x4d(weights=pretrained)
    return ResNet(model=model, n_blocks=n_blocks)


@BACKBONES.register_module
def resnext50_32x4d_ssl(pretrained=None, n_blocks=5):
    model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext50_32x4d_ssl')
    return ResNet(model=model, n_blocks=n_blocks)


@BACKBONES.register_module
def resnext101_32x8d(pretrained=None, n_blocks=5):
    model = resnet.resnext101_32x8d(weights=pretrained)
    return ResNet(model=model, n_blocks=n_blocks)

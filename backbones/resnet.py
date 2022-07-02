from torch import nn
from torchvision.models import resnet
from backbones._base import BackboneBase
from builder import BACKBONES


@BACKBONES.register_module
class ResNet(BackboneBase):
    def __init__(self, model_name='resnet50', pretrained=False, n_blocks=5, init_config=None):
        super(ResNet, self).__init__(init_config=init_config)
        model = get_model(model_name, pretrained)

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
        self.out_channels = tuple([3, block1_channel] + [block[-1].conv1.in_channels for block in blocks[1:n_blocks]])

    def forward(self, x):
        features = [x]
        for block in self.blocks.values():
            x = block(x)
            features.append(x)
        return features


def get_model(model_name, pretrained):
    model = None
    if model_name == 'resnet18':
        model = resnet.resnet18(pretrained=pretrained)
    elif model_name == 'resnet34':
        model = resnet.resnet34(pretrained=pretrained)
    elif model_name == 'resnet50':
        model = resnet.resnet50(pretrained=pretrained)
    elif model_name == 'resnet101':
        model = resnet.resnet101(pretrained=pretrained)
    elif model_name == 'resnet152':
        model = resnet.resnet152(pretrained=pretrained)
    elif model_name == 'resnext50_32x4d':
        model = resnet.resnext50_32x4d(pretrained=pretrained)
    elif model_name == 'resnext101_32x8d':
        model = resnet.resnext101_32x8d(pretrained=pretrained)
    else:
        NotImplementedError(f'{model_name}')
    return model


if __name__ == '__main__':
    import torch
    _model = ResNet('resnet50', pretrained=False, stage_index=5)

    with torch.no_grad():
        feats = _model(torch.rand(4, 3, 224, 224))

    print(_model.out_channels)
    for feat in feats:
        print(feat.shape)

from torch import nn
from torchvision.models import resnet
from backbones._base import BackboneBase
from builder import BACKBONES


@BACKBONES.register_module
class ResNet(BackboneBase):
    def __init__(self, model_name='resnet50', pretrained=False, init_config=None):
        super(ResNet, self).__init__(init_config=init_config)
        model = get_model(model_name, pretrained)

        self.stem = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu
        )
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        self._in_channels = (
            model.conv1.out_channels,
            model.layer1[-1].conv1.in_channels,
            model.layer2[-1].conv1.in_channels,
            model.layer3[-1].conv1.in_channels,
            model.layer4[-1].conv1.in_channels,
        )

    def forward(self, x):
        feats = []
        x = self.stem(x)
        feats.append(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        feats.append(x)
        x = self.layer2(x)
        feats.append(x)
        x = self.layer3(x)
        feats.append(x)
        x = self.layer4(x)
        feats.append(x)
        return feats


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
        assert 'no model_name'
    return model


if __name__ == '__main__':
    import torch
    _model = ResNet('resnet50', pretrained=False)

    with torch.no_grad():
        feats = _model(torch.rand(4, 3, 224, 224))
    for feat in feats:
        print(feat.shape)

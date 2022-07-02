from torch import nn
import timm
from builder import BACKBONES


@BACKBONES.register_module
class EfficientNet(nn.Module):
    def __init__(self, model_name='efficientnet_b0', pretrained=False, n_blocks=5, drop_path_rate=0.1):
        super(EfficientNet, self).__init__()
        base_net = timm.create_model(model_name=model_name, pretrained=pretrained, drop_path_rate=drop_path_rate)
        blocks = [
            nn.Sequential(base_net.conv_stem, base_net.bn1, *base_net.blocks[0]),
            base_net.blocks[1],
            base_net.blocks[2],
            nn.Sequential(*base_net.blocks[3:5]),
            base_net.blocks[5:]
        ]

        self.blocks = nn.ModuleDict({
            f'block{index}': block for index, block in zip(range(n_blocks), blocks)
        })

        block1_channel = blocks[0][-1].conv_pw.out_channels
        block2_channel = blocks[1][-1].conv_pwl.out_channels
        block3_channel = blocks[2][-1].conv_pwl.out_channels
        out_channels = [block[-1][-1].conv_pwl.out_channels for block in blocks[3:n_blocks]]
        self.out_channels = tuple([3, block1_channel, block2_channel, block3_channel] + out_channels)
    
    def forward(self, x):
        features = [x]
        for block in self.blocks.values():
            x = block(x)
            features.append(x)
        return features


if __name__ == '__main__':
    import torch
    image = torch.rand(2, 3, 224, 224)
    model = EfficientNet(model_name='efficientnet_b5', stage_index=5).eval()

    with torch.no_grad():
        feats = model(image)

    for feat in feats:
        print(feat.shape)


from torch import nn
import timm
from builder import BACKBONES


@BACKBONES.register_module
class EfficientNet(nn.Module):
    def __init__(self, model_name='efficientnet_b2', pretrained=False, stage_index=5):
        super(EfficientNet, self).__init__()
        base_net = timm.create_model(model_name=model_name, pretrained=pretrained, drop_path_rate=0.1)
        self.stem = base_net.conv_stem
        self.bn1 = base_net.bn1
        blocks = [
            base_net.blocks[0],
            base_net.blocks[1],
            base_net.blocks[2],
            base_net.blocks[3:5],
            base_net.blocks[5:]
        ]

        self.blocks = nn.ModuleList(blocks[:stage_index])
    
    def forward(self, x):
        feats = []
        x = self.bn1(self.stem(x))
        for block in self.blocks:
            x = block(x)
            feats.append(x)
        
        return feats

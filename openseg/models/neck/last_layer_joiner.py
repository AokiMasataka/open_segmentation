from torch import nn
from .pos_embed import PositionEmbeddingSineHW
from ..builder import NECKS


@NECKS.register_module
class LastLayerJoinner(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, pos_embed: bool = False):
        super(LastLayerJoinner, self).__init__()
        self.joint = nn.Linear(in_features=input_dim, out_features=output_dim, bias=True)
        self.pos_embed = PositionEmbeddingSineHW()
    
    def forward(self, features):
        feature = features[-1]
        B, C, H, W = feature.shape
        feature = self.pos_embed(feature) + feature
        feature = feature.view(B, C, W * H).transpose(1, 2)
        return self.joint(feature)
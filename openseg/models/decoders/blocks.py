from torch import nn
from openbacks.layers import SelfAttention, FeedForward


__all__ = ('ResBlock', 'CenterBlock', 'TransformerCenterBlock')


class ResBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        layers: int = 1,
        eps: float = 1e-6,
        activation: str = 'relu',
        dropout: float = 0.0,
        downsample: bool = False,
        upsample: bool = False
    ):
        super(ResBlock, self).__init__()
        assert (not downsample) or (not upsample)
        blocks = list()
        for _ in range(layers):
            blocks.append(
                ResLayer(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    eps=eps,
                    activation=activation,
                    dropout=dropout
                )
            )
        
        self.blocks = nn.Sequential(*blocks)
        
        self.downsample = downsample
        self.upsample = upsample
        self.sampling = None
        if downsample:
            self.sampling = nn.Conv2d(out_dim, out_dim, kernel_size=(3, 3), stride=(2, 2), padding=1)
        
        if upsample:
            upsample = upsample if isinstance(upsample, int) else 2
            self.sampling = nn.Sequential(
                nn.Conv2d(out_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
                nn.Upsample(scale_factor=upsample, mode='bilinear')
            )
    
    def forward(self, x):
        x = self.blocks(x)
        if self.sampling is not None:
            x = self.sampling(x)
        return x


class ResLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, eps: float = 1e-6, activation: str = 'relu', dropout: float = 0.0):
        super(ResLayer, self).__init__()
        activations = {'relu': nn.ReLU, 'gelu': nn.GELU}
        assert activation in activations.keys()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(out_dim, eps=eps),
            activations[activation](),
            nn.Dropout2d(dropout),
            nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(out_dim, eps=eps),
            activations[activation](),
        )

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(out_dim, out_dim // 16, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim // 16, out_dim, kernel_size=(1, 1), stride=(1, 1)),
            nn.Sigmoid()
        )

        self.is_shrotcat = in_dim != out_dim
        self.conv_shrotcat = None
        if self.is_shrotcat:
            self.conv_shrotcat = nn.Conv2d(in_dim, out_dim, kernel_size=(1, 1), stride=(1, 1), padding=0)

    
    def forward(self, x):
        shrotcat = x
        x = self.block(x)
        x = x * self.se(x)

        if self.is_shrotcat:
            return x + self.conv_shrotcat(shrotcat)
        else:
            return x + shrotcat


class CenterBlock(nn.Sequential):
    def __init__(self, in_dim):
        conv1 = nn.Conv2d(
            in_dim,
            in_dim,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1,
            bias=False
        )
        bn1 = nn.BatchNorm2d(in_dim)
        activate1 = nn.ReLU()

        conv2 = nn.Conv2d(
            in_dim,
            in_dim,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1,
            bias=False
        )
        bn2 = nn.BatchNorm2d(in_dim)
        activate2 = nn.ReLU()

        super().__init__(conv1, bn1, activate1, conv2, bn2, activate2)


class TransformerLayer(nn.Module):
    def __init__(self, in_dim:  int, num_heads: int, dropout: float = 0.0, eps: float = 1e-6):
        super(TransformerLayer, self).__init__()

        self.attention = SelfAttention(dim_query=in_dim, num_heads=num_heads, dropout=dropout)
        self.attention_norm = nn.LayerNorm(in_dim, eps=eps)
        self.ffn = FeedForward(dim=in_dim, dropout=dropout, activation_fn='gelu')
        self.ffn_norm = nn.LayerNorm(in_dim, eps=eps)
    
    def forward(self, x):
        x = self.attention(self.attention_norm(x)) + x
        x = self.ffn(self.ffn_norm(x)) + x
        return x


class TransformerCenterBlock(nn.Module):
    def __init__(self, in_dim: int, num_heads: int, num_layers: int = 1, dropout: float = 0.0, eps: float = 1e-6):
        super(TransformerCenterBlock, self).__init__()
        layers = list()
        for i in range(num_layers):
            layers.append(TransformerLayer(in_dim, num_heads, dropout, eps))

        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        B, C, W, H = x.shape
        x = x.view(B, C, W * H).transpose(1, 2)

        x = self.layers(x)

        x = x.transpose(1, 2).view(B, C, W, H)
        return x

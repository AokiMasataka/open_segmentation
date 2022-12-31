import torch
from torch import nn
from openbacks.layers import CrossAttention, SelfAttention, FeedForward


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
        upsample: bool = False,
        extend: int = 1,
    ):
        super(ResBlock, self).__init__()
        assert (not downsample) or (not upsample)
        blocks = list()
        for i in range(layers):
            in_dim = in_dim if i == 0 else out_dim
            blocks.append(
                ResLayer(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    eps=eps,
                    activation=activation,
                    dropout=dropout,
                    extend=extend
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
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        eps: float = 1e-6,
        activation: str = 'relu',
        dropout: float = 0.0,
        extend: int = 1,
    ):
        super(ResLayer, self).__init__()
        activations = {'relu': nn.ReLU, 'gelu': nn.GELU, 'silu': nn.SiLU, 'swish': nn.SiLU}
        assert activation in activations.keys()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=out_dim * extend, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(out_dim * extend, eps=eps),
            activations[activation](),
            nn.Dropout2d(dropout),
            nn.Conv2d(in_channels=out_dim * extend, out_channels=out_dim, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(out_dim, eps=eps),
            activations[activation](),
        )

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(out_dim, out_dim // 8, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim // 8, out_dim, kernel_size=(1, 1), stride=(1, 1)),
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
    def __init__(self, in_dim, eps=1e-5, activation_fn='relu'):
        activation_fns = {'relu': nn.ReLU, 'gelu': nn.GELU, 'silu': nn.SiLU}
        super().__init__(
            nn.Conv2d(in_dim, in_dim, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(in_dim, eps=eps),
            activation_fns[activation_fn](),
            nn.Conv2d(in_dim, in_dim, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(in_dim, eps=eps),
            activation_fns[activation_fn](),
        )


class TransformerEncoderLayer(nn.Module):
    def __init__(self, in_dim:  int, num_heads: int, dropout: float = 0.0, eps: float = 1e-5):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = SelfAttention(embed_dim=in_dim, num_heads=num_heads, dropout=dropout, relative_pos_bias=True)
        self.attention_norm = nn.LayerNorm(in_dim, eps=eps)
        self.ffn = FeedForward(input_dim=in_dim, dropout=dropout, activation_fn='gelu')
        self.ffn_norm = nn.LayerNorm(in_dim, eps=eps)
    
    def forward(self, x):
        x = self.attention(self.attention_norm(x)) + x
        x = self.ffn(self.ffn_norm(x)) + x
        return x


class BasicTransformerLayer(nn.Module):
    def __init__(self, embed_dim, cross_dim, num_heads, dropout=0.0, eps=1e-5):
        super(BasicTransformerLayer, self).__init__()
        self.cross_attn = CrossAttention(embed_dim=embed_dim, cross_dim=cross_dim, num_heads=num_heads, dropout=dropout, relative_pos_bias=True)
        self.cross_norm = nn.LayerNorm(embed_dim, eps=eps)
        self.self_attn = SelfAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, relative_pos_bias=True)
        self.self_norm = nn.LayerNorm(embed_dim, eps=eps)
        self.ffn = FeedForward(input_dim=embed_dim, dropout=dropout, activation_fn='gelu')
        self.ffn_norm = nn.LayerNorm(embed_dim, eps=eps)
    
    def forward(self, hidden_state, context):
        hidden_state = self.cross_attn(self.cross_norm(hidden_state), context) + hidden_state
        hidden_state = self.self_attn(self.self_norm(hidden_state)) + hidden_state
        hidden_state = self.ffn(self.ffn_norm(hidden_state))
        return hidden_state


class TransformerCenterBlock(nn.Module):
    def __init__(self, in_dim: int, num_heads: int, num_layers: int = 1, dropout: float = 0.0, eps: float = 1e-5):
        super(TransformerCenterBlock, self).__init__()
        layers = list()
        for _ in range(num_layers):
            layers.append(TransformerEncoderLayer(in_dim, num_heads, dropout, eps))

        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        B, C, W, H = x.shape
        x = x.view(B, C, W * H).transpose(1, 2)
        x = self.layers(x)
        x = x.transpose(1, 2).view(B, C, W, H)
        return x


class BasicTransformer2D(nn.Module):
    def __init__(self, embed_dim, cross_dim, num_heads, num_layers, dropout=0.0, eps=1e-5):
        super(BasicTransformer2D, self).__init__()
        blocks = [BasicTransformerLayer(
            embed_dim=embed_dim, cross_dim=cross_dim, num_heads=num_heads, dropout=dropout, eps=eps
        ) for _ in range(num_layers)]
        self.blocks = nn.ModuleList(blocks)
    
    def forward(self, hidden_state, context):
        B, C, W, H = hidden_state.shape
        hidden_state = hidden_state.view(B, C, W * H).transpose(1, 2)

        for block in self.blocks:
            hidden_state = block(hidden_state, context)
        
        return hidden_state.transpose(1, 2).view(B, C, W, H)

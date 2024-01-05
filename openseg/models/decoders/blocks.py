from torch import nn
from openback.layer import SEModule, CrossAttention, SelfAttention, FeedForward


_ACTIVATE = dict(relu=nn.ReLU, gelu=nn.GELU, silu=nn.SELU)


class ResLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: str = 'relu',
        dropout: float = 0.0,
        eps: float = 1e-5,
        extend: int = 1,
    ):
        super(ResLayer, self).__init__()
        assert activation in _ACTIVATE.keys()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=output_dim * extend, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(output_dim * extend, eps=eps),
            _ACTIVATE[activation](),
            nn.Dropout2d(dropout),
            nn.Conv2d(in_channels=output_dim * extend, out_channels=output_dim, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(output_dim, eps=eps),
        )

        self.se = SEModule(channels=output_dim, rd_ratio=1 / 8)

        self.is_shrotcat = input_dim != output_dim
        self.conv_shrotcat = None
        if self.is_shrotcat:
            self.conv_shrotcat = nn.Conv2d(input_dim, output_dim, kernel_size=(1, 1), stride=(1, 1), padding=0)
        
        self.out_activate = _ACTIVATE[activation](),
    
    def forward(self, x):
        shrotcat = x
        x = self.block(x)
        x = self.se(x)
        if self.is_shrotcat:
            return self.out_activate(x + self.conv_shrotcat(shrotcat))
        else:
            return self.out_activate(x + shrotcat)


class ResBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        layers: int = 1,
        downsample: bool = False,
        upsample: bool = False,
        activation: str = 'relu',
        dropout: float = 0.0,
        eps: float = 1e-5,
        extend: int = 1,
    ):
        super(ResBlock, self).__init__()
        assert (not downsample) or (not upsample)

        blocks = list()
        for i in range(layers):
            input_dim = input_dim if i == 0 else output_dim
            blocks.append(
                ResLayer(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    activation=activation,
                    dropout=dropout,
                    eps=eps,
                    extend=extend
                )
            )
        
        self.downsample = downsample
        self.upsample = upsample
        self.sampling = None
        if downsample:
            blocks.append(nn.Conv2d(output_dim, output_dim, kernel_size=(3, 3), stride=(2, 2), padding=1))
        elif upsample:
            upsample = upsample if isinstance(upsample, int) else 2
            blocks.append(nn.Upsample(scale_factor=upsample, mode='nearest'))
        
        self.blocks = nn.Sequential(*blocks)
    
    def forward(self, x):
        return self.blocks(x)


class UpsampleBlock(nn.Sequential):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_layers: int,
        scale: int = 2,
        mode: str = 'nearest',
        activation: str = 'relu',
        dropout: float = 0.0,
        eps: float = 1e-5,
    ):
        super(UpsampleBlock, self).__init__()
        for i in range(num_layers):
            input_dim = input_dim if i == 0 else output_dim
            self.append(nn.Conv2d(
                in_channels=input_dim,
                out_channels=output_dim,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1,
                bias=False
            ))
            self.append(nn.BatchNorm2d(num_features=output_dim, eps=eps))
            self.append(_ACTIVATE[activation]())
        self.append(nn.Dropout2d(p=dropout))
        self.append(SEModule(channels=output_dim, rd_ratio=1 / 8))
        if scale > 0:
            self.append(nn.Upsample(scale_factor=scale, mode=mode))


class CenterBlock(nn.Sequential):
    def __init__(self, input_dim: int, activation: str ='relu', eps: float = 1e-5):
        super(CenterBlock, self).__init__(
            nn.Conv2d(input_dim, input_dim, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(input_dim, eps=eps),
            _ACTIVATE[activation](),
            nn.Conv2d(input_dim, input_dim, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(input_dim, eps=eps),
            _ACTIVATE[activation](),
        )


class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_dim:  int, num_heads: int, dropout: float = 0.0, eps: float = 1e-5):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = SelfAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout, relative_pos_bias=True)
        self.attention_norm = nn.LayerNorm(input_dim, eps=eps)
        self.ffn = FeedForward(input_dim=input_dim, dropout=dropout, activation_fn='gelu')
        self.ffn_norm = nn.LayerNorm(input_dim, eps=eps)
    
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
    def __init__(self, input_dim: int, num_heads: int, num_layers: int = 1, dropout: float = 0.0, eps: float = 1e-5):
        super(TransformerCenterBlock, self).__init__()
        layers = list()
        for _ in range(num_layers):
            layers.append(TransformerEncoderLayer(input_dim, num_heads, dropout, eps))

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
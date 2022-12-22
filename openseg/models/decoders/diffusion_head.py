import math
import torch
from torch import nn
from .base import DecoderBase
from .blocks import ResLayer, BasicTransformer2D


class DiffusionHead(DecoderBase):
    # https://data-analytics.fun/2022/08/24/diffusion-model-pytorch-1/
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        context_dim: int = 512,
        encoder_dims: tuple = (64, 128, 256, 512), # TODO channel Mapper
        block_out_dims: tuple = (64, 128, 256, 512),
        num_heads_par_layers: tuple = (1, 2, 4, 8),
        upblock_types: tuple = ('C', 'T', 'T', 'T'),
        downblock_types: tuple = ('T', 'T', 'T', 'C'),
        depths: tuple = (2, 1, 1, 1),
        attention_depths: tuple = (1, 1, 1, 1),
        dropout: float = 0.0,
        init_config: dict = None
    ):
        super(DiffusionHead, self).__init__(init_config=init_config)
        assert len(block_out_dims) == len(num_heads_par_layers) == len(upblock_types) == len(downblock_types)
        num_layers = len(block_out_dims)
        downblock_types = downblock_types[::-1]
        self.block_out_dims = block_out_dims
        self.context_dim = context_dim

        self.time_embed = nn.Sequential(
            PositionEmbeddings(embed_dim=block_out_dims[0]),
            nn.Linear(in_features=block_out_dims[0], out_features=block_out_dims[-1]),
            nn.SiLU(),
            nn.Linear(in_features=block_out_dims[-1], out_features=block_out_dims[-1]),
            nn.SiLU()
        )
        
        self.downblocks = nn.ModuleList()
        self.upblocks = nn.ModuleList()
        next_dim = block_out_dims[1]
        prov_dim = input_dim
        block_out_dims = block_out_dims + (0,)

        for i in range(num_layers):
            dwom_attention_module = True if upblock_types[i] == 'T' else False
            up_attention_module = True if downblock_types[i] == 'T' else False
            scale = 4 if i == 0 else 2

            block_out_dim = block_out_dims[i]
            num_heads = num_heads_par_layers[i]
            depth = depths[i]
            attention_depth = attention_depths[i]
            next_dim = block_out_dims[i + 1]

            downblock = DiffusionBlock(
                embed_dim=prov_dim,
                output_dim=block_out_dim,
                time_embed_dim=block_out_dims[-2],
                context_dim=context_dim,
                num_heads=num_heads,
                num_layers=depth,
                num_layers_attn=attention_depth,
                attn_drop=dropout,
                conv_drop=dropout,
                downsampling=scale,
                attention_module=dwom_attention_module
            )

            upblock = DiffusionBlock(
                embed_dim=next_dim + block_out_dim,
                output_dim=num_classes if i == 0 else block_out_dim,
                time_embed_dim=block_out_dims[-2],
                context_dim=context_dim,
                num_heads=num_heads,
                num_layers=depth,
                num_layers_attn=attention_depth,
                attn_drop=dropout,
                conv_drop=dropout,
                upsampling=scale,
                attention_module=up_attention_module
            )

            self.downblocks.append(downblock)
            self.upblocks.append(upblock)

            prov_dim = block_out_dim

        self.upblocks = self.upblocks[::-1]
    
    def forward(self, hidden_state, context, timestep):
        assert timestep.ndim == 1
        time_embeds = self.time_embed(timestep)

        cat_states = list()
        for downblock in self.downblocks:
            hidden_state = downblock(hidden_state=hidden_state, context=context, temb=time_embeds)
            cat_states.append(hidden_state)
        
        cat_states = cat_states[::-1]
        cat_states[0] = None
        
        for upblock, cat_state in zip(self.upblocks, cat_states):
            if cat_state is not None:
                hidden_state = torch.cat((hidden_state, cat_state), dim=1)
            hidden_state = upblock(hidden_state=hidden_state, context=context, temb=time_embeds)
        
        return hidden_state


class DiffusionBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        output_dim: int,
        time_embed_dim: int,
        context_dim: int,
        num_heads: int,
        num_layers: int,
        num_layers_attn: int,
        attn_drop: float = 0.0,
        conv_drop: float = 0.0,
        upsampling: int = False,
        downsampling: int = False,
        attention_module: bool = True,
    ):
        super(DiffusionBlock, self).__init__()
        assert not (upsampling and downsampling)
        self.output_dim = output_dim
        self.attention_module = attention_module

        self.time_embed_proj = nn.Linear(time_embed_dim, output_dim, bias=True)

        self.attentions = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        for i in range(num_layers):
            embed_dim = embed_dim if i == 0 else output_dim
            self.resblocks.append(
                ResLayer(in_dim=embed_dim, out_dim=output_dim, eps=1e-6, activation='swish', dropout=conv_drop, extend=2)
            )
            if attention_module:
                self.attentions.append(
                    BasicTransformer2D(
                        embed_dim=output_dim,
                        dim_cross=context_dim,
                        num_heads=num_heads,
                        num_layers=num_layers_attn,
                        dropout=attn_drop
                    )
                )
            else:
                self.attentions.append(nn.Identity())
        
        if upsampling:
            self.sampling = nn.Sequential(
                nn.Upsample(scale_factor=upsampling, mode='bilinear'),
                nn.Conv2d(in_channels=output_dim, out_channels=output_dim, kernel_size=(1, 1), stride=(1, 1))
            )
        
        if downsampling:
            if downsampling == 2:
                self.sampling = nn.Conv2d(output_dim, output_dim, kernel_size=(3, 3), stride=(downsampling, downsampling), padding=(1, 1))
            else:
                kernel = (downsampling, downsampling)
                self.sampling = nn.Conv2d(output_dim, output_dim, kernel_size=kernel, stride=kernel, padding=0)
    
    def forward(self, hidden_state, context, temb):
        for resblock, attention in zip(self.resblocks, self.attentions):
            hidden_state = resblock(hidden_state) + self.time_embed_proj(temb).view(-1, self.output_dim, 1, 1)
            if self.attention_module:
                hidden_state = attention(hidden_state=hidden_state, context=context)
        
        hidden_state = self.sampling(hidden_state)
        return hidden_state


class PositionEmbeddings(nn.Module):
  def __init__(self, embed_dim):
    super(PositionEmbeddings, self).__init__()
    half_dim = embed_dim // 2
    embeddings = math.log(10000) / (half_dim - 1)
    embeddings = torch.exp(torch.arange(half_dim) * -embeddings)
    embeddings = embeddings.unsqueeze(0)
    self.register_buffer(name='embeddings', tensor=embeddings)
 
  def forward(self, time):
    embeddings = time.unsqueeze(1) * self.embeddings
    embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=1)
    return embeddings

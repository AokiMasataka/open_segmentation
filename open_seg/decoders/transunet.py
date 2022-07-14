import math
import torch
from torch import nn
import timm
from timm.models.layers import trunc_normal_

from ._base import DecoderBase, DECODER_BLOCK
from open_seg.builder import DECODERS


@DECODERS.register_module
class TransUnet(DecoderBase):
	def __init__(self, encoder_channels, decoder_channels, transformer_name, transformer_image_size, block_type='basic'):
		super(TransUnet, self).__init__()
		assert block_type in DECODER_BLOCK.keys()

		self._encoder_channels = encoder_channels
		self._decoder_channels = decoder_channels
		encoder_channels = list(encoder_channels)
		decoder_channels = list(decoder_channels)

		self.transformer = Transformer(
			model_name=transformer_name,
			image_size=transformer_image_size,
			input_dim=encoder_channels[-1]
		)

		encoder_channels = [*encoder_channels, self.transformer.embed_dim]
		encoder_channels.reverse()
		# computing blocks input and output channels
		head_channels = encoder_channels[0]
		in_channels = [head_channels] + decoder_channels[:-1]
		skip_channels = encoder_channels[1:] + [0]
		out_channels = decoder_channels
		in_channels = [in_ch + skip_ch for in_ch, skip_ch in zip(in_channels, skip_channels)]

		block = DECODER_BLOCK[block_type]
		blocks = [
			block(in_ch, out_ch) for in_ch, out_ch in zip(in_channels, out_channels)
		]
		self.blocks = nn.ModuleList(blocks)
		# transformer name: 'vit_base_patch16_384'

	def forward(self, features):
		features.reverse()
		x = self.transformer(features[0])

		for decoder_block, feature in zip(self.blocks, features):
			x = decoder_block(x=x, skip=feature)

		x = self.blocks[-1](x=x)
		return x

	def decoder_out_dim(self):
		return self._decoder_channels[-1]


class Transformer(nn.Module):
	def __init__(self, model_name, input_dim, image_size, patch_size=2):
		super(Transformer, self).__init__()
		_base = timm.create_model(model_name=model_name, pretrained=True)
		patch_size_t2 = (patch_size, patch_size)

		self.patch_embed = nn.Conv2d(input_dim, _base.embed_dim, kernel_size=patch_size_t2, stride=patch_size_t2)
		num_patches = (image_size // patch_size) * (image_size // patch_size)
		
		self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, _base.embed_dim))
		self.pos_drop = _base.pos_drop
		self.blocks = _base.blocks
		self.norm = _base.norm
		self.embed_dim = _base.embed_dim

		trunc_normal_(self.pos_embed, std=0.02)
	
	def forward(self, hidden_state):
		hidden_state = self.patch_embed(hidden_state)
		hidden_state = hidden_state.flatten(2).transpose(1, 2)

		hidden_state = self.pos_drop(hidden_state + self.pos_embed)
		hidden_state = self.blocks(hidden_state)
		hidden_state = self.norm(hidden_state)
		seq = int(math.sqrt(hidden_state.shape[1]))
		hidden_state = hidden_state.transpose(1, 2).view(-1, self.embed_dim, seq, seq)
		return hidden_state

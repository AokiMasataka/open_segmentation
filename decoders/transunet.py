from torch import nn
import timm
from timm.models.layers import trunc_normal_
from decoders.base import DecoderBasicBlock, DecoderBottleneckBlock
from builder import DECODERS


@DECODERS.register_module
class TransUnet(nn.Module):
	def __init__(self, encoder_channels, decoder_channels, transformer_name, transformer_image_size, block_type='basic'):
		super(TransUnet, self).__init__()
		self._encoder_channels = encoder_channels
		self._decoder_channels = decoder_channels

		encoder_channels = encoder_channels[::-1]

		# computing blocks input and output channels
		head_channels = encoder_channels[0]
		in_channels = [head_channels] + list(decoder_channels[:-1])
		skip_channels = list(encoder_channels[1:]) + [0]
		out_channels = decoder_channels

		# combine decoders keyword arguments
		block = DecoderBasicBlock
		if block_type == 'basic':
			block = DecoderBasicBlock
		elif block_type == 'bottleneck':
			block = DecoderBottleneckBlock
		else:
			ValueError(f'block_type is basic or bottleneck')

		blocks = [
			block(in_ch, skip_ch, out_ch) for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
		]
		self.blocks = nn.ModuleList(blocks)
		# transformer name: 'vit_base_patch16_384'
		self.transformer = Transformer(
			model_name=transformer_name,
			image_size=transformer_image_size,
			input_dim=encoder_channels[0]
		)

	def forward(self, *features):
		features = features[::-1]  # reverse channels to start from head of encoder

		x = features[0]
		x = self.transformer(x)
		skips = features[1:]

		for i, decoder_block in enumerate(self.blocks):
			skip = skips[i] if i < len(skips) else None
			x = decoder_block(x, skip)
		return x


class Transformer(nn.Module):
	def __init__(self, model_name, input_dim, image_size, patch_size=1):
		super(Transformer, self).__init__()
		_base = timm.create_model(model_name=model_name, pretrained=True)

		if _base.embed_dim != input_dim:
			self.patch_embed = nn.Linear(in_features=input_dim, out_features=_base.embed_dim)
		else:
			self.patch_embed = None
		num_patches = (image_size // patch_size) * (image_size // patch_size)
		
		self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, _base.embed_dim))
		self.pos_drop = _base.pos_drop
		self.blocks = _base.blocks
		self.norm = _base.norm

		trunc_normal_(self.pos_embed, std=0.02)
	
	def forward(self, feat):
		feat = feat.flatten(2).transpose(1, 2)
		if self.patch_embed is not None:
			feat = self.patch_embed(feat)

		feat = self.pos_drop(feat + self.pos_embed)
		feat = self.blocks(feat)
		feat = self.norm(feat)
		return feat


if __name__ == '__main__':
	import torch

	_feat = torch.rand(2, 3, 24, 24)
	_model = Transformer(model_name='vit_base_patch16_384', input_dim=3, image_size=24)

	with torch.no_grad():
		_feat = _model(_feat)

	print(_feat.shape)

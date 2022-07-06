from torch import nn
import timm
from open_seg.backbones._base import BackboneBase
from open_seg.builder import BACKBONES


class ConvNeXt(BackboneBase):
	def __init__(self, base_net, stage_index=5, init_config=None):
		super(ConvNeXt, self).__init__(init_config)
		blocks = [self.stem, *base_net.stages]

		self.blocks = nn.ModuleDict({
			f'block{index}': block for index, block in zip(range(stage_index), blocks)
		})

	def forward(self, x):
		features = []
		for block in self.blocks.values():
			x = block(x)
			features.append(x)
		return features


@BACKBONES.register_module
def convnext_tiny(pretrained=False, drop_path_rate=0.0, stage_index=5, stem_type='dual', init_config=None):
	base_net = timm.create_model(
		model_name='convnext_tiny',
		pretrained=pretrained,
		drop_path_rate=drop_path_rate,
		stem_type=stem_type
	)
	return ConvNeXt(base_net=base_net, stage_index=stage_index, init_config=init_config)


@BACKBONES.register_module
def convnext_small(pretrained=False, drop_path_rate=0.0, stage_index=5, stem_type='dual', init_config=None):
	base_net = timm.create_model(
		model_name='convnext_small',
		pretrained=pretrained,
		drop_path_rate=drop_path_rate,
		stem_type=stem_type
	)
	return ConvNeXt(base_net=base_net, stage_index=stage_index, init_config=init_config)


@BACKBONES.register_module
def convnext_base(pretrained=False, drop_path_rate=0.0, stage_index=5, stem_type='dual', init_config=None):
	base_net = timm.create_model(
		model_name='convnext_base',
		pretrained=pretrained,
		drop_path_rate=drop_path_rate,
		stem_type=stem_type
	)
	return ConvNeXt(base_net=base_net, stage_index=stage_index, init_config=init_config)


@BACKBONES.register_module
def convnext_large(pretrained=False, drop_path_rate=0.0, stage_index=5, stem_type='dual', init_config=None):
	base_net = timm.create_model(
		model_name='convnext_large',
		pretrained=pretrained,
		drop_path_rate=drop_path_rate,
		stem_type=stem_type
	)
	return ConvNeXt(base_net=base_net, stage_index=stage_index, init_config=init_config)

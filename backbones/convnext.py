from torch import nn
import timm
# from builder import BACKBONES


# BACKBONES.register_module
class ConvNeXt(nn.Module):
	def __init__(self, model_name, pretrained=False, drop_path_rate=0.0, stage_index=5, stem_type='dual'):
		super(ConvNeXt, self).__init__()
		base_net = timm.create_model(
			model_name=model_name,
			pretrained=pretrained,
			drop_path_rate=drop_path_rate,
			stem_type=stem_type
		)
		self.stem = base_net.stem
		self.stages = base_net.stages[:stage_index - 1]
	
	def forward(self, x):
		feats = []
		x = self.stem(x)
		feats.append(x)
		for stage in self.stages:
			x = stage(x)
			feats.append(x)
		return feats


if __name__ == '__main__':
	import torch
	image = torch.rand(2, 3, 224, 224)
	model = ConvNeXt(model_name='convnext_small', drop_path_rate=0.1, stage_index=4)

	with torch.no_grad():
		feats = model(image)
	
	for feat in feats:
		print(feat.shape)

from copy import deepcopy
from openbacks import build_backbone


class Registry:
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def register_module(self, module, module_name=None):
        self._module_dict[module.__name__] = module
        return module

    def build(self, config):
        if isinstance(config, (list, tuple)):
            return {conf['type']: self.build(config=deepcopy(conf)) for conf in config}
        else:
            _type = config.pop('type')
            module = self._module_dict[_type]
            return module(**config)

    def get_module(self, name):
        return self._module_dict[name]


BACKBONES = Registry(name='backbones')
DECODERS = Registry(name='decoders')
LOSSES = Registry(name='losses')
SEGMENTER = Registry(name='segmenters')


def build_model(config):
    try:
        backbone = BACKBONES.build(config=config['backbone'])
    except:
        backbone = build_backbone(config=config['backbone'])
    try:
        config['decoder']['encoder_channels'] = backbone.out_channels()
        config['decoder']['scale_factors'] = backbone.scale_factors
    except:
        pass
    decoder = DECODERS.build(config=config['decoder'])
    losses = LOSSES.build(config=config['loss'])
    segmenter_module = SEGMENTER.get_module(config['segmenter']['type'])
    model = segmenter_module(
        backbone=backbone,
        decoder=decoder,
        losses=losses,
        num_classes=config['num_classes'],
        init_config=config.get('init_config', None),
        test_config=config.get('test_config', None),
        norm_config=config.get('norm_config', None)
    )
    return model

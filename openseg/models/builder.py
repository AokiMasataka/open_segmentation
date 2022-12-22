from copy import deepcopy


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
    config['init_config'] = config.get('init_config', None)
    config['test_config'] = config.get('test_config', None)
    config['norm_config'] = config.get('norm_config', None)
    segmenter_module = SEGMENTER.get_module(config.pop('type'))
    model = segmenter_module(**config)
    return model


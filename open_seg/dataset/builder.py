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


PIPELINES = Registry(name='pipeline')


def build_pipeline(config):
    pipe_dict = PIPELINES.build(config=config)
    compose_module = PIPELINES.get_module(name='Compose')
    return compose_module(transforms=pipe_dict)

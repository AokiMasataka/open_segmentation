from copy import deepcopy


class Registry:
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()
    
    @property
    def name(self):
        return self._name
    
    @property
    def module_dict(self):
        return self._module_dict
    
    def __getitem__(self, item):
        return self._module_dict[item]

    def register_module(self, module, module_name=None):
        if module_name is None:
            self._module_dict[module.__name__] = module
        else:
            self._module_dict[module_name] = module
        return module

    def build(self, config):
        if isinstance(config, (list, tuple)):
            return {conf['type']: self.build(config=deepcopy(conf)) for conf in config}
        else:
            _type = config.pop('type')
            module = self._module_dict[_type]
            return module(**config)

    def get_module(self, name):
        return self._module_dict.get(name, None)

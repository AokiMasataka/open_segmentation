from copy import deepcopy
from .get_params import get_params_beckbone, get_params_decoder, get_params_seg_head


class Registry:
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def register_module(self, module, module_name=None):
        if module_name is None:
            module_name = module.__name__
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
        return self._module_dict[name]


OPTIMIZERS = Registry(name='optimizers')
SCHEDULERS = Registry(name='scheduler')


def build_optimizer(module, config):
    base_lr = config.pop('base_lr')
    if 'head_lr' in config.keys():
        head_lr = config.pop('head_lr')
    else:
        head_lr = base_lr
    weight_decay = config.pop('weight_decay')
    params = get_params_beckbone(module=module, lr=base_lr, weight_decay=weight_decay)
    params += get_params_decoder(module=module, lr=head_lr, weight_decay=weight_decay)
    params += get_params_seg_head(module=module, lr=head_lr, weight_decay=weight_decay)
    config['params'] = params
    optimizer = OPTIMIZERS.build(config=config)
    return optimizer


def build_scheduler(optimizer_module, config):
    config['optimizer'] = optimizer_module
    scheduler = SCHEDULERS.build(config=config)
    return scheduler

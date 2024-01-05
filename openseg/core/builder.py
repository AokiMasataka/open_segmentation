from copy import deepcopy
from .get_params import get_params, get_params_beckbone, get_params_decoder
from openback.utils import Registry


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
    # params = get_params(module=module, base_lr=base_lr, head_lr=head_lr, weight_decay=weight_decay)
    params = module.parameters()
    config['params'] = params
    optimizer = OPTIMIZERS.build(config=config)
    return optimizer


def build_scheduler(optimizer_module, config):
    config['optimizer'] = optimizer_module
    scheduler = SCHEDULERS.build(config=config)
    return scheduler

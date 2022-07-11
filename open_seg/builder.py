import torch


class Registry:
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def register_module(self, module, module_name=None):
        self._module_dict[module.__name__] = module
        return module

    # def build(self, config):
    #     _type = config.pop('type')
    #     module = self._module_dict[_type]
    #     _instance = module(**config)
    #     return _instance

    # def build_losses(self, config):
    #     losses = {}
    #     if isinstance(config, dict):
    #         config_list = [config]
    #     else:
    #         config_list = config
    #
    #     for config in config_list:
    #         _type = config.pop('type')
    #         module = self._module_dict[_type]
    #         losses[_type] = module(**config)
    #     return losses

    # def build_pipline(self, config, mode='train_pipeline'):
    #     compose_module = self._module_dict['Compose']
    #
    #     pipelines = {}
    #     config = config[mode]
    #     if not isinstance(config, list):
    #         config_list = [config]
    #     else:
    #         config_list = config
    #
    #     for config in config_list:
    #         _type = config.pop('type')
    #         module = self._module_dict[_type]
    #         pipelines[_type] = module(**config)
    #
    #     pipeline = compose_module(pipelines)
    #     return pipeline

    def build_pipline(self, config, mode='train_pipeline'):
        compose_module = self._module_dict['Compose']

        config = config[mode]
        pipelines = self.build(config=config)

        pipeline = compose_module(pipelines)
        return pipeline

    def build(self, config):
        if isinstance(config, (list, tuple)):
            return {conf['type']: self.build(config=conf) for conf in config}
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
OPTIMIZER = Registry(name='optimizers')
PIPELINES = Registry(name='pipelines')


def build_backbone(config):
    return BACKBONES.build(config)


def build_decoder(config):
    return DECODERS.build(config)


def build_segmenter(config):
    return SEGMENTER.build(config)


def build_losses(config):
    return LOSSES.build(config)


def build_optimizer(config):
    return OPTIMIZER.build(config)

def build_scheduler(config):
    if config['T_max'] == 'total_step':
        config['T_max'] = config['total_step']
    del config['total_step']
    return OPTIMIZER.build(config)


def build_pipeline(config, mode='train_pipeline'):
    return PIPELINES.build_pipline(config, mode=mode)


def build_model(config):
    backbone = build_backbone(config['backbone'])
    config['decoder']['encoder_channels'] = backbone.out_channels
    decoder = build_decoder(config['decoder'])
    losses = build_losses(config['loss'])
    module = SEGMENTER.get_module(config['segmenter']['type'])
    model = module(backbone=backbone, decoder=decoder, losses=losses, num_classes=config['segmenter']['num_classes'])
    init_config = config.get('init_config', None)
    if init_config is not None:
        model.load_state_dict(torch.load(init_config['weight_path']))
    return model

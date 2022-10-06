def get_params(model, config):
    no_decay = ('bias', 'LayerNorm.bias', 'LayerNorm.weight', 'BatchNorm2d.bias', 'BatchNorm2d.weight')

    params = [
        {
            'params': [p for n, p in model.backbone.named_parameters() if any(nd in n for nd in no_decay)],
            'lr': config['base_lr'],
            'weight_decay': 0.0
        },
        {
            'params': [p for n, p in model.backbone.named_parameters() if not any(nd in n for nd in no_decay)],
            'lr': config['base_lr'],
            'weight_decay': config['weight_decay']
        },
        {
            'params': [p for n, p in model.decoder.named_parameters() if any(nd in n for nd in no_decay)],
            'lr': config['head_lr'],
            'weight_decay': 0.0
        },
        {
            'params': [p for n, p in model.decoder.named_parameters() if not any(nd in n for nd in no_decay)],
            'lr': config['head_lr'],
            'weight_decay': config['weight_decay']
        },
        {
            'params': [p for n, p in model.seg_head.named_parameters() if any(nd in n for nd in no_decay)],
            'lr': config['head_lr'],
            'weight_decay': 0.0
        },
        {
            'params': [p for n, p in model.seg_head.named_parameters() if not any(nd in n for nd in no_decay)],
            'lr': config['head_lr'],
            'weight_decay': config['weight_decay']
        }
    ]
    return params


def get_params_beckbone(module, lr, weight_decay):
    no_decay = ('bias', 'LayerNorm.bias', 'LayerNorm.weight', 'BatchNorm2d.bias', 'BatchNorm2d.weight')

    params = [
        {
            'params': [p for n, p in module.backbone.named_parameters() if any(nd in n for nd in no_decay)],
            'lr': lr,
            'weight_decay': 0.0
        },
        {
            'params': [p for n, p in module.backbone.named_parameters() if not any(nd in n for nd in no_decay)],
            'lr': lr,
            'weight_decay': weight_decay
        },
    ]
    return params


def get_params_decoder(module, lr, weight_decay):
    no_decay = ('bias', 'LayerNorm.bias', 'LayerNorm.weight', 'BatchNorm2d.bias', 'BatchNorm2d.weight')

    params = [
        {
            'params': [p for n, p in module.decoder.named_parameters() if any(nd in n for nd in no_decay)],
            'lr': lr,
            'weight_decay': 0.0
        },
        {
            'params': [p for n, p in module.decoder.named_parameters() if not any(nd in n for nd in no_decay)],
            'lr': lr,
            'weight_decay': weight_decay
        },
    ]
    return params


def get_params_seg_head(module, lr, weight_decay):
    no_decay = ('bias', 'LayerNorm.bias', 'LayerNorm.weight', 'BatchNorm2d.bias', 'BatchNorm2d.weight')

    params = [
        {
            'params': [p for n, p in module.seg_head.named_parameters() if any(nd in n for nd in no_decay)],
            'lr': lr,
            'weight_decay': 0.0
        },
        {
            'params': [p for n, p in module.seg_head.named_parameters() if not any(nd in n for nd in no_decay)],
            'lr': lr,
            'weight_decay': weight_decay
        },
    ]
    return params

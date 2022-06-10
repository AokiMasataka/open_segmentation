def get_params(module, base_lr, head_lr, weight_decay):
    no_decay = ('bias', 'LayerNorm.bias', 'LayerNorm.weight', 'BatchNorm2d.bias', 'BatchNorm2d.weight')

    params = [
        {
            'params': [p for n, p in module.named_parameters() if any(nd in n for nd in no_decay) and 'decoder' not in n],
            'lr': base_lr,
            'weight_decay': 0.0
        },
        {
            'params': [p for n, p in module.named_parameters() if not any(nd in n for nd in no_decay)  and 'decoder' not in n],
            'lr': base_lr,
            'weight_decay': weight_decay
        },
        {
            'params': [p for n, p in module.named_parameters() if any(nd in n for nd in no_decay) and 'decoder' in n],
            'lr': head_lr,
            'weight_decay': 0.0
        },
        {
            'params': [p for n, p in module.named_parameters() if not any(nd in n for nd in no_decay)  and 'decoder' in n],
            'lr': head_lr,
            'weight_decay': weight_decay
        },
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

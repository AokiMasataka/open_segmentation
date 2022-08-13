import torch
from open_seg.builder import OPTIMIZER


__all__ = ['Adam', 'AdamW']


def add_weight_decay(model):
    all_params = set(model.parameters())
    wd_params = set()
    for m in model.modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
            wd_params.add(m.weight)
    no_wd_params = all_params - wd_params
    return list(no_wd_params), list(wd_params)


def weight_decay_group(module):
    decay_params = []
    no_decay_params = []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith('.bias') or name in 'pos_embed':
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return no_decay_params, decay_params


class DefaultOptimizerConstructor:
    def __init__(self, base_lr, weight_decay):
        self.base_lr = base_lr
        self.weight_decay = weight_decay

    def __call__(self, module):
        return module.parameters()


class WeightDecayOptimizerConstructor:
    def __init__(self, base_lr, weight_decay):
        self.base_lr = base_lr
        self.weight_decay = weight_decay

    def __call__(self, module):
        parameter_groups = {}

        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights

            if len(param.shape) == 1 or name.endswith('.bias') or name in 'pos_embed':
                group_name = 'no_decay'
                weight_decay = 0.0
            else:
                group_name = 'decay'
                weight_decay = self.weight_decay

            parameter_groups[group_name] = {
                'params': [],
                'weight_decay': weight_decay,
                'param_names': [],
                'group_name': group_name,
                'lr': self.base_lr,
            }

            parameter_groups[group_name]['params'].append(param)
            parameter_groups[group_name]['param_names'].append(name)
        return parameter_groups.values()


class GroupLROptimizerConstructor:
    def __init__(self, base_lr, groups_lr, weight_decay):
        self.base_lr = base_lr
        self.groups_lr = groups_lr
        self.weight_decay = weight_decay

    def __call__(self, module):
        parameter_groups = {}
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith('.bias') or name in 'pos_embed':
                group_name = 'no_decay'
                weight_decay = 0.0
            else:
                group_name = 'decay'
                weight_decay = self.weight_decay

            lr, layer_group = None, None
            for key, value in self.groups_lr.items():
                if key in name:
                    lr = value
                    layer_group = key
                else:
                    continue

            if layer_group is None:
                lr = self.base_lr
                layer_group = 'no_group'

            group_name = f'layer_{layer_group.replace(".", "_")}_{group_name}'

            if group_name not in parameter_groups:
                print(f'INFO - GroupLROptimizerConstructor: name: {name} - lr: {lr} - weight decay: {weight_decay}')
                parameter_groups[group_name] = {
                    'params': [],
                    'weight_decay': weight_decay,
                    'param_names': [],
                    'group_name': group_name,
                    'lr': self.base_lr,
                }
            parameter_groups[group_name]['params'].append(param)
            parameter_groups[group_name]['param_names'].append(name)
        return parameter_groups.values()


# class Adam:
#     def __init__(self):
#         pass
#
#     @staticmethod
#     def build(module, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0, amsgrad=False, constructor='DefaultOptimizerConstructor'):
#         parameter_groups =
#         return

@OPTIMIZER.register_module
def Adam(model, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0, amsgrad=False):
    # no_decay, decay = add_weight_decay(model)
    no_decay, decay = weight_decay_group(model)

    params = [{'params': no_decay, 'weight_decay': 0.0}, {'params': decay}]

    optimizer = torch.optim.Adam(
        params=params,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        amsgrad=amsgrad,
    )
    return optimizer


@OPTIMIZER.register_module
def AdamW(model, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False):
    no_decay, decay = add_weight_decay(model)

    params = [{'params': no_decay, 'weight_decay': 0.0}, {'params': decay}]

    optimizer = torch.optim.AdamW(
        params=params,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        amsgrad=amsgrad,
    )
    return optimizer

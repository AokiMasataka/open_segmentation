import copy


def load_config_file(path):
    with open(path, 'r') as f:
        text = f.read()
    config = dict()
    exec(text, globals(), config)

    if '_base_' in config.keys():
        for base_path in config['_base_']:
            config = substitute_base_vars(config=config, base_var_dict=config, base_cfg=load_config_file(base_path))

    text = str(config)
    return config, text


def substitute_base_vars(config, base_var_dict, base_cfg):
    """Substitute variable strings to their actual values."""
    config = copy.deepcopy(config)

    if isinstance(config, dict):
        for k, v in config.items():
            if isinstance(v, str) and v in base_var_dict:
                new_v = base_cfg
                for new_k in base_var_dict[v].split('.'):
                    new_v = new_v[new_k]
                config[k] = new_v
            elif isinstance(v, (list, tuple, dict)):
                config[k] = substitute_base_vars(v, base_var_dict, base_cfg)
    elif isinstance(config, tuple):
        config = tuple(substitute_base_vars(c, base_var_dict, base_cfg) for c in config)
    elif isinstance(config, list):
        config = [substitute_base_vars(c, base_var_dict, base_cfg) for c in config]
    elif isinstance(config, str) and config in base_var_dict:
        new_v = base_cfg
        for new_k in base_var_dict[config].split('.'):
            new_v = new_v[new_k]
        config = new_v
    return config

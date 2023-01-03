def load_config_file(path):
    with open(path, 'r') as f:
        text = f.read()
    
    config = dict()
    exec(text, globals(), config)

    if '_base_' in config.keys():
        for base_config_path in config.pop('_base_'):
            base_config = load_config_file(path=base_config_path)
            config = merge_configs(config=config, base_config=base_config)

    return config


def merge_configs(config, base_config):
    if isinstance(base_config, dict):
        for key in base_config.keys():
            if key in config.keys():
                # print(config[key])
                config[key] = merge_configs(config=config[key], base_config=base_config[key])
            else:
                config[key] = base_config[key]
    else:
        return config
    return config
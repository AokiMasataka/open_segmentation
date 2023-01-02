def load_config_file(path):
    with open(path, 'r') as f:
        text = f.read()
    
    config = dict()
    exec(text, globals(), config)

    if '_base_' in config.keys():
        for base_config_path in config['_base_']:
            base_config, _ = load_config_file(path=base_config_path)
            config = merge_configs(config=config, base_config=base_config)

    text = ''
    for key, value in config.items():
        text += str(key) + '=' + str(value) + '\n'
    return config, text


def merge_configs(config, base_config):
    if isinstance(base_config, dict):
        for key in base_config.keys():
            if key in config.keys():
                config[key] = merge_configs(config=config[key], base_config=base_config[key])
            else:
                config[key] = base_config[key]
    else:
        return base_config
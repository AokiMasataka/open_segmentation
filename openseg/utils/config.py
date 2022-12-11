import copy


def load_config_file(path):
    with open(path, 'r') as f:
        text = f.read()
    config = dict()
    exec(text, globals(), config)

    # if '_base_' in config.keys():
    #     for base_path in config['_base_']:
    #         config = substitute_base_vars(config=config, base_var_dict=config, base_cfg=load_config_file(base_path))

    text = str(config)
    return config, text

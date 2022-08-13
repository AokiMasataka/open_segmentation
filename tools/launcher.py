import os
import sys
import warnings
from open_seg.api import trainner


def load_config_file(path):
    with open(path, 'r') as f:
        text = f.read()
    config = dict()
    exec(text, globals(), config)

    if '_base_' in config.keys():
        for base_path in config['_base_']:
            base_config, _ = load_config_file(path=base_path)
            config = config | base_config

    text = str(config)
    return config, text


def main():
    warnings.simplefilter('ignore', UserWarning)
    config_file = sys.argv[1]
    config, text = load_config_file(config_file)
    os.makedirs(config['work_dir'], exist_ok=True)
    with open(config['work_dir'] + '/config.py', 'w') as f:
        f.write(text)

    trainner(config=config)


if __name__ == '__main__':
    main()

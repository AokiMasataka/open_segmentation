import os
import warnings
import argparse
from openseg import TrainnerArgs, train, load_config_file


def main():
    warnings.simplefilter('ignore', UserWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='path to config file')
    parser.add_argument('-d', '--device', type=str, default='cuda', help='select device')
    parser.add_argument('--log_level', type=str, default='info', help='set log level info or debug')

    args = parser.parse_args()
    config, config_text = load_config_file(path=args.config)
    
    os.makedirs(config['work_dir'], exist_ok=True)
    with open(config['work_dir'] + '/config.py', 'w') as f:
        f.write(config_text)
    
    trainner_args = TrainnerArgs(
        **config['train_config'],
        work_dir=config['work_dir'],
        log_level=args.log_level,
        device=args.device
    )
    train(config=config, trainner_args=trainner_args)


if __name__ == '__main__':
    main()

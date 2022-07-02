import os
import sys
import warnings
from torch.utils.data import DataLoader

from api import train
from dataset import SegmentData, train_collate_fn
from utils import seed_everything, get_logger

import builder


def load_config_file(path):
    with open(path, 'r') as f:
        text = f.read()
    config = dict()
    exec(text, globals(), config)
    return config, text


def main():
    warnings.simplefilter('ignore', UserWarning)
    config_file = sys.argv[1]
    config, text = load_config_file(config_file)
    os.makedirs(config['work_dir'], exist_ok=True)
    seed_everything(config['train_config'].get('seed', 0))
    logger = get_logger(log_file=config['work_dir'] + '/train.log', stream=True)

    train_dataset = SegmentData(**config['data']['train'], test_mode=False)
    valid_dataset = SegmentData(**config['data']['valid'], test_mode=True)
    train_pipeline = builder.build_pipeline(config, mode='train_pipeline')
    valid_pipeline = builder.build_pipeline(config, mode='test_pipeline')
    train_dataset.get_pipeline(train_pipeline)
    valid_dataset.get_pipeline(valid_pipeline)
    train_laoder = DataLoader(
        dataset=train_dataset,
        batch_size=config['data']['batch_size'],
        num_workers=config['data'].get('num_workers', os.cpu_count()),
        shuffle=True,
        pin_memory=True,
        collate_fn=train_collate_fn
    )

    logger.info('successful build datasets')

    model = builder.build_model(config)
    model.cuda()
    logger.info('successful build model')

    optimizer_config = config['optimizer']
    optimizer_config['model'] = model
    _optimizer = builder.build_optimizer(optimizer_config)

    scheduler_config = config['scheduler']
    scheduler_config['optimizer'] = _optimizer
    scheduler_config['total_step'] = config['train_config'].get('max_iters', 50000)
    scheduler = builder.build_scheduler(scheduler_config)

    logger.info('successful build optimizer')

    logger.info('train start')

    with open(config['work_dir'] + '/config.py', 'w') as f:
        f.write(text)

    train(
        model=model,
        optimizer=_optimizer,
        lr_scheduler=scheduler,
        train_laoder=train_laoder,
        valid_dataset=valid_dataset,
        logger=logger,
        train_config=config['train_config'],
        save_dir=config['work_dir'],
    )


if __name__ == '__main__':
    main()

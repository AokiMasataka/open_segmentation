import os
import sys
import torch
from torch.utils.data import DataLoader

from dataset import SegmentData
from api import train
from utils import seed_everything, get_logger

import backbones
import decoders
import losses
import segmenter
import dataset
import optimizer
import builder


def load_config_file(path):
    with open(path, 'r') as f:
        text = f.read()
    config = dict()
    exec(text, globals(), config)
    return config, text


def main():
    config_file = sys.argv[1]
    config, text = load_config_file(config_file)
    os.makedirs(config['work_dir'], exist_ok=True)
    seed_everything(config['seed'])
    logger = get_logger(log_file=config['work_dir'] + '/train.log', stream=True)

    train_dataset = SegmentData(**config['data']['train'], test_mode=False)
    valid_dataset = SegmentData(**config['data']['valid'], test_mode=True)
    train_pipeline = builder.build_pipeline(config, mode='train_pipeline')
    valid_pipeline = builder.build_pipeline(config, mode='valid_pipeline')
    train_dataset.get_pipeline(train_pipeline)
    valid_dataset.get_pipeline(valid_pipeline)
    train_laoder = DataLoader(
        dataset=train_dataset,
        batch_size=config['data']['batch_size'],
        num_workers=config['data'].get('num_workers', os.cpu_count()),
        shuffle=True,
        pin_memory=True
    )
    total_step = train_laoder.__len__() * config['data']['epochs']
    logger.info('INFO: successful build datasets')

    model = builder.build_model(config)
    model.cuda()
    logger.info('INFO: successful build model')

    optimizer_config = config['optimizer']
    optimizer_config['model'] = model
    _optimizer = builder.build_optimizer(optimizer_config)

    scheduler_config = config['scheduler']
    scheduler_config['optimizer'] = _optimizer
    scheduler_config['total_step'] = total_step
    scheduler = builder.build_scheduler(scheduler_config)

    logger.info('INFO: successful build optimizer')

    logger.info('INFO: train start')

    with open(config['work_dir'] + '/config.py', 'w') as f:
        f.write(text)

    train(
        model=model,
        optimizer=_optimizer,
        lr_scheduler=scheduler,
        train_laoder=train_laoder,
        valid_dataset=valid_dataset,
        logger=logger,
        epochs=config['data']['epochs'],
        threshold=0.5,
        fp16=config.get('fp16', False),
        save_dir=config['work_dir'],
        tta=config['tta'],
    )


if __name__ == '__main__':
    main()

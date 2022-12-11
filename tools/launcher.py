import os
import json
import warnings
import argparse
from torch.utils.data import DataLoader

from openseg.dataset import CustomDataset, InfiniteSampler, train_collate_fn
from openseg.models import build_model
from openseg.core import build_optimizer, build_scheduler, DummyScheduler
from openseg.engin import TrainnerArgs, trainner
from openseg.utils import load_config_file


def build_components(config):
    data_config = config['data']
    train_dataset = CustomDataset(**data_config['train'])
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=data_config['batch_size'],
        sampler=InfiniteSampler(dataset_size=train_dataset.__len__()),
        num_workers=data_config.get('num_workers', os.cpu_count()),
        pin_memory=True,
        collate_fn=train_collate_fn
    )
    valid_dataset = CustomDataset(**data_config['valid'])
    model = build_model(config=config['model'])
    optimizer = build_optimizer(module=model, config=config['optimizer'])
    scheduler_config = config.get('scheduler', False)
    if scheduler_config:
        scheduler = build_scheduler(optimizer_module=optimizer, config=scheduler_config)
    else:
        scheduler = DummyScheduler()

    return model, optimizer, scheduler, train_loader, valid_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='path to config file')
    parser.add_argument('-d', '--device', type=str, default='cuda', help='select device')
    parser.add_argument('--log_level', type=str, default='info', help='set log level info or debug')

    args = parser.parse_args()

    warnings.simplefilter('ignore', UserWarning)
    config_file = args.config
    config, text = load_config_file(config_file)
    os.makedirs(config['work_dir'], exist_ok=True)
    with open(config['work_dir'] + '/config.py', 'w') as f:
        f.write(text)

    print(json.dumps(config, indent=4))

    model, optimizer, scheduler, train_loader, valid_dataset = build_components(config=config)
    model.to(args.device)

    trainner_args = TrainnerArgs(
        **config['train_config'],
        work_dir=config['work_dir'],
        log_level=args.log_level,
        device=args.device
    )

    trainner(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        valid_dataset=valid_dataset,
        trainner_args=trainner_args
    )


if __name__ == '__main__':
    main()

import os
import sys
from copy import deepcopy
import itertools

import torch
from torch.cuda import amp
from torch.utils.data import DataLoader

from open_seg.dataset import SegmentData, train_collate_fn
from open_seg.utils import seed_everything, get_logger
from open_seg import builder
from .valid import valid_fn


class DummyScheduler:
    def __init__(self):
        pass

    def step(self):
        pass


class InfiniteSampler:
    def __init__(self, dataset_size):
        self.dataset_size = dataset_size
        self.generator = torch.Generator(device='cpu')

    def __iter__(self):
        yield from itertools.islice(self._infinite(), 0, None, 1)

    def _infinite(self):
        while True:
            yield from torch.randperm(self.dataset_size, generator=self.generator)


def train_one_step(model, optimizer, lr_scheduler, batch, scaler, fp16):
    images, labels = batch['image'].cuda(), batch['label'].cuda()
    optimizer.zero_grad()
    with amp.autocast(enabled=fp16):
        loss, losses = model.forward_train(image=images, label=labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    lr_scheduler.step()
    return loss, losses


def trainner(config):
    seed_everything(config['train_config'].get('seed', 0))
    logger = get_logger(log_file=config['work_dir'] + '/train.log', stream=True)

    logger.info(f'Python info: {sys.version}')
    logger.info(f'PyTroch version: {torch.__version__}')
    logger.info(f'GPU model: {torch.cuda.get_device_name(0)}')

    data_config = config['data']
    train_dataset = SegmentData(**data_config['train'], test_mode=False)
    valid_dataset = SegmentData(**data_config['valid'], test_mode=True)
    train_pipeline = builder.build_pipeline(config=config, mode='train_pipeline')
    valid_pipeline = builder.build_pipeline(config=config, mode='test_pipeline')
    train_dataset.get_pipeline(train_pipeline)
    valid_dataset.get_pipeline(valid_pipeline)
    train_laoder = DataLoader(
        dataset=train_dataset,
        batch_size=data_config['batch_size'],
        sampler=InfiniteSampler(dataset_size=train_dataset.__len__()),
        num_workers=data_config.get('num_workers', os.cpu_count()),
        pin_memory=True,
        collate_fn=train_collate_fn
    )

    logger.info('successful build datasets')

    model = builder.build_model(config=config['model'])
    model.cuda()
    logger.info('successful build model')

    optimizer_config = config['optimizer']
    optimizer_config['model'] = model
    optimizer = builder.build_optimizer(config=optimizer_config)

    scheduler_config = config.get('scheduler', None)

    if scheduler_config is not None:
        scheduler_config['optimizer'] = optimizer
        scheduler_config['total_step'] = config['train_config'].get('max_iters', 50000)
        lr_scheduler = builder.build_scheduler(config=scheduler_config)
    else:
        lr_scheduler = DummyScheduler()

    logger.info('successful build optimizer')

    train_config = config['train_config']
    max_iters = train_config.get('max_iters', 50_000)
    eval_interval = train_config.get('eval_interval', 5_000)
    save_checkpoint = train_config.get('save_checkpoint', False)
    log_interval = train_config.get('log_interval', 500)
    threshold = train_config.get('threshold', 0.5)
    checkpoint = train_config.get('checkpoint', False)
    save_dir = config['work_dir']
    weight_name = train_config.get('weight_name', '')

    best_loss = float('Inf')
    best_score = -float('Inf')
    best_state = deepcopy(model.state_dict())
    mean_train_loss = 0.0
    loss_dict = {key: 0.0 for key in model.losses.keys()}

    os.makedirs(f'{save_dir}/checkpoint/', exist_ok=True)
    if checkpoint:
        cpt = torch.load(checkpoint)
        model.load_state_dict(cpt['model'])
        optimizer.load_state_dict(cpt['optimizer'])
        lr_scheduler.load_state_dict(cpt['lr_scheduler'])
        start_step = cpt['step'] + 1
    else:
        start_step = 1

    fp16 = train_config.get('fp16', False)
    logger.info(f'use amp: {fp16}')
    scaler = amp.GradScaler(enabled=fp16)
    for step, batch in zip(range(start_step, max_iters + 1), train_laoder):
        loss, losses = train_one_step(
            model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, batch=batch, scaler=scaler, fp16=fp16
        )

        mean_train_loss += loss.item()
        for key in losses.keys():
            loss_dict[key] += losses[key].item()

        if step % log_interval == 0:
            loss_log = ''
            for key, value in zip(loss_dict.keys(), loss_dict.values()):
                loss_log += f' - {key}: {value / log_interval:.6f}'
            loss_log += f' - loss: {mean_train_loss / log_interval:.6f}'

            last_lr = lr_scheduler.get_last_lr()
            lr_log = f' - lr: {last_lr[0]:.6f}'

            logger.info(f'step: [{step}/{max_iters}]{lr_log}{loss_log}')
            mean_train_loss = 0.0
            loss_dict = {key: 0.0 for key in model.losses.keys()}

        if step % eval_interval == 0:
            if valid_dataset is not None:
                model.eval()
                valid_score = valid_fn(model=model, dataset=valid_dataset, threshold=threshold)
                logger.info(f'step: [{step}/{max_iters}] - dice score: {valid_score:.6f}')
                if best_score < valid_score:
                    best_score = valid_score
                    best_state = deepcopy(model.state_dict())
                model.train()
            else:
                if (mean_train_loss / log_interval) < best_loss:
                    best_loss = mean_train_loss / log_interval
                    best_score = 0.0
                    best_state = deepcopy(model.state_dict())

            if save_checkpoint:
                cpt = {
                    'model': deepcopy(model.state_dict()),
                    'optimizer': deepcopy(optimizer.state_dict()),
                    'lr_scheduler': deepcopy(lr_scheduler.state_dict()),
                    'step': step
                }
                torch.save(cpt, f'{save_dir}/checkpoint/step{step}{weight_name}.cpt')

    # logger.info(f'Best loss: {best_loss:.6f}')
    logger.info(f'Best score: {best_score:.6f}')
    torch.save(best_state, f'{save_dir}/checkpoint/best_score{weight_name}.pth')
    torch.save(model.state_dict(), f'{save_dir}/checkpoint/last{weight_name}.pth')

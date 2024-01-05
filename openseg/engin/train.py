import os
import sys
import copy
import logging
import warnings
import torch
from torch.cuda import amp
from torch.utils.data import DataLoader
from .evalate import evalate
from ..dataset import build_dataset, InfiniteSampler
from ..models import build_segmentor
from ..core import DummyScheduler, build_optimizer, build_scheduler
from openback.utils import set_logger, seed_everything


def build_data_loader(config: dict) -> DataLoader:
    dataset = build_dataset(config=config['dataset'])
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config['batch_size'],
        num_workers=config.get('num_workers', os.cpu_count()),
        collate_fn=dataset.train_collate_fn,
        sampler=InfiniteSampler(dataset_size=dataset.__len__()),
    )
    return data_loader


def build_train_componet(config: dict):
    train_loader = build_data_loader(config=config['data']['train'])
    valid_loader = build_data_loader(config=config['data']['valid'])
    
    model = build_segmentor(config=config['model'])
    optimizer = build_optimizer(module=model, config=config['optimizer'])
    scheduler_config = config.get('scheduler', False)
    if scheduler_config:
        scheduler = build_scheduler(optimizer_module=optimizer, config=scheduler_config)
    else:
        scheduler = DummyScheduler()

    return model, optimizer, scheduler, train_loader, valid_loader


class TrainnerArgs:
    def __init__(
            self,
            max_iters: int = None,
            eval_interval: int = None,
            epochs: int = None,
            log_interval: int = None,
            work_dir: str = 'dummy string',
            seed: int = 42,
            load_checkpoint: str = None,
            save_checkpoint: str = False,
            fp16: bool = True,
            threshold: float = 0.5,
            log_level: str = 'info',
            device='cuda'
    ):
        assert max_iters is not None or epochs is not None
        assert log_interval is not None
        assert work_dir != 'dummy string'

        self.seed = seed
        self.max_iters = max_iters
        self.epochs = epochs
        self.log_interval = log_interval
        self.load_checkpoint = load_checkpoint
        self.save_checkpoint = save_checkpoint
        self.fp16 = fp16
        self.threshold = threshold
        self.log_level = log_level
        self.work_dir = work_dir
        self.device = device

        if eval_interval is None:
            self.eval_interval = max_iters // 10
        else:
            self.eval_interval = eval_interval


def train(config: dict, trainner_args: TrainnerArgs):
    warnings.simplefilter('ignore')
    log_path = os.path.join(trainner_args.work_dir, 'train.log')
    set_logger(log_file=log_path, level=trainner_args.log_level)
    seed_everything(seed=trainner_args.seed)

    logging.info(f'Python info: {sys.version}')
    logging.info(f'PyTroch version: {torch.__version__}')
    logging.info(f'GPU model: {torch.cuda.get_device_name(0)}')
    logging.info(msg=f'use amp: {trainner_args.fp16}')

    model, optimizer, scheduler, train_loader, valid_dataset = build_train_componet(config=config)
    model.to(trainner_args.device)

    one_epoch_size = train_loader.dataset.__len__() // train_loader.batch_size
    best_loss = float('Inf')
    best_score = -float('Inf')
    best_state = copy.deepcopy(model.state_dict())
    mean_train_loss = 0.0
    loss_dict = {key: 0.0 for key in model.losses.keys()}

    if trainner_args.save_checkpoint:
        os.makedirs(f'{trainner_args.work_dir}/checkpoints/', exist_ok=True)
    
    if trainner_args.load_checkpoint:
        cpt = torch.load(trainner_args.load_checkpoint)
        model.load_state_dict(cpt['model'])
        optimizer.load_state_dict(cpt['optimizer'])
        scheduler.load_state_dict(cpt['lr_scheduler'])
        start_step = cpt['step'] + 1
        logging.info(msg=f'checkpoint load from: {trainner_args.load_checkpoint}')
    else:
        start_step = 1

    scaler = amp.GradScaler(enabled=trainner_args.fp16)

    # for step, batch in zip(range(start_step, trainner_args.max_iters + 1), itertools.cycle(train_loader)):
    for step, batch in zip(range(start_step, trainner_args.max_iters + 1), train_loader):
        logging.debug(msg=f'step: {step}')

        images, labels = batch['images'].to(trainner_args.device), batch['labels'].to(trainner_args.device)
        optimizer.zero_grad()
        with amp.autocast(enabled=trainner_args.fp16):
            loss, losses = model.forward_train(images=images, labels=labels)
        
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()

        mean_train_loss += loss.item()
        for key in losses.keys():
            loss_dict[key] += losses[key].item()
        
        if step % trainner_args.log_interval == 0:
            log_msg = f'epoch: {step/one_epoch_size:.2f} - step: [{step}/{trainner_args.max_iters}]'
            log_msg += f' - lr: {scheduler.get_last_lr()[0]:.6f}'

            for key, value in zip(loss_dict.keys(), loss_dict.values()):
                log_msg += f' - {key}: {value / trainner_args.log_interval:.6f}'
            log_msg += f' - loss: {mean_train_loss / trainner_args.log_interval:.6f}'

            logging.info(msg=log_msg)
            mean_train_loss = 0.0
            loss_dict = {key: 0.0 for key in model.losses.keys()}
        
        if step % trainner_args.eval_interval == 0:
            if valid_dataset is not None:
                model.eval()
                valid_scores = evalate(
                    model=model,
                    valid_loader=valid_dataset,
                    metrics=config['metrics'],
                    fp16=trainner_args.fp16,
                    device=trainner_args.device
                )

                valid_acc = valid_scores.pop('accuracy')
                evalate_msg = f' - accuracy: {valid_acc:.4f}'

                for key, value in valid_scores.items():
                    evalate_msg += f' - {key}: {value:.6f}'
                
                logging.info(msg=f'epoch: {step/one_epoch_size:.2f} - step: [{step}/{trainner_args.max_iters}]' + evalate_msg)
                if best_score < valid_acc:
                    best_score = valid_acc
                    best_state = copy.deepcopy(model.state_dict())
                model.train()
            else:
                if (mean_train_loss / trainner_args.log_interval) < best_loss:
                    best_loss = mean_train_loss / trainner_args.log_interval
                    best_score = 0.0
                    best_state = copy.deepcopy(model.state_dict())

            if trainner_args.save_checkpoint:
                cpt = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': scheduler.state_dict(),
                    'step': step
                }
                torch.save(cpt, f'{trainner_args.work_dir}/checkpoints/step{step}_checkpoint.cpt')
                del cpt
        
    logging.info(f'Best score: {best_score:.6f}')
    torch.save(best_state, os.path.join(trainner_args.work_dir, 'best_score.pth'))
    torch.save(model.state_dict(), os.path.join(trainner_args.work_dir, 'last.pth'))
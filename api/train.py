import os
from copy import deepcopy
import torch
from torch.cuda import amp

from api.valid import valid_fn


def train(
        model,
        optimizer,
        lr_scheduler,
        train_laoder,
        valid_dataset,
        logger,
        train_config,
        save_dir='',
    ):
    max_iters = train_config.get('max_iters', 50_000)
    eval_interval = train_config.get('eval_interval', 5_000)
    save_checkpoint = train_config.get('save_checkpoint', False)
    log_interval = train_config.get('log_interval', 500)
    threshold = train_config.get('threshold', 0.5)
    checkpoint = train_config.get('checkpoint', False)

    best_loss = float('Inf')
    best_score = -float('Inf')
    best_state = deepcopy(model.state_dict())
    mean_train_loss = 0.0
    loss_dict = {key: 0.0 for key in model.losses.keys()}

    if checkpoint:
        cpt = torch.load(checkpoint)
        model.load_state_dict(cpt['model'])
        optimizer.load_state_dict(cpt['optimizer'])
        lr_scheduler.load_state_dict(cpt['lr_scheduler'])
        start_step = cpt['step']
    else:
        start_step = 0

    fp16 = train_config.get('fp16', False)
    logger.info(f'use amp: {fp16}')
    scaler = amp.GradScaler(enabled=fp16)

    step = start_step
    while True:
        for batch in train_laoder:
            step += 1
            image, label = batch['image'].cuda(), batch['label'].cuda()
            optimizer.zero_grad()

            with amp.autocast(enabled=fp16):
                loss, losses = model.forward_train(image=image, label=label)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            lr_scheduler.step()

            mean_train_loss += loss.item()
            for key in losses.keys():
                loss_dict[key] += losses[key].item()

            if (step % eval_interval == 0 and step % log_interval == 0) or step % eval_interval == 0:
                if valid_dataset is not None:
                    model.eval()
                    valid_score = valid_fn(model=model, dataset=valid_dataset, threshold=threshold)
                    logger.info(f'iter: [{step}] - dice score: {valid_score:.6f}')
                    if best_score < valid_score:
                        best_score = valid_score
                        best_state = deepcopy(model.state_dict())
                    model.train()
                else:
                    if (mean_train_loss / log_interval) < best_loss:
                        best_loss = mean_train_loss / log_interval
                        best_score = 0.0
                        best_state = deepcopy(model.state_dict())

                mean_train_loss = 0.0
                loss_dict = {key: 0.0 for key in model.losses.keys()}

                if save_checkpoint:
                    os.makedirs(f'{save_dir}/checkpoint/', exist_ok=True)
                    cpt = {
                        'model': deepcopy(model.state_dict()),
                        'optimizer': deepcopy(optimizer.state_dict()),
                        'lr_scheduler': deepcopy(lr_scheduler.state_dict()),
                        'step': step
                    }
                    torch.save(cpt, f'{save_dir}/checkpoint/step{step}.cpt')

            elif step % log_interval == 0:
                loss_log = ''
                for key, value in zip(loss_dict.keys(), loss_dict.values()):
                    loss_log += f' - {key}: {value / log_interval:.6f}'
                loss_log += f' - loss: {mean_train_loss / log_interval:.6f}'

                last_lr = lr_scheduler.get_last_lr()
                lr_log = f' - lr: {last_lr:.6f}'

                logger.info(f'step: [{step}/{max_iters}]{lr_log}{loss_log}')
                mean_train_loss = 0.0
                loss_dict = {key: 0.0 for key in model.losses.keys()}

            if step == max_iters:
                break
        if step == max_iters:
            break

    logger.info(f'Best loss: {best_loss:.6f}')
    logger.info(f'Best score: {best_score:.6f}')
    torch.save(best_state, f'{save_dir}/best_loss.pth')
    torch.save(model.state_dict(), f'{save_dir}/last.pth')

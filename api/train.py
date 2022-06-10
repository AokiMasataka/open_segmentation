from copy import deepcopy
import numpy as np
import torch
from torch.cuda import amp

from api.valid import valid, valid_tta
import matplotlib.pyplot as plt


def train(
        model,
        optimizer,
        lr_scheduler,
        train_laoder,
        valid_dataset,
        logger,
        epochs,
        threshold=0.5,
        fp16=False,
        save_dir='',
        tta=False
    ):

    step_per_epoch = train_laoder.__len__()
    log_interval = step_per_epoch // 10

    best_loss = float('Inf')
    best_score = -float('Inf')
    best_state = deepcopy(model.state_dict())

    logger.info(f'INFO: use amp {fp16}')
    scaler = amp.GradScaler(enabled=fp16)
    for epoch in range(epochs):
        model.train()

        train_mean_loss = 0.0

        for step, batch in enumerate(train_laoder, 1):
            image, label = batch['image'].cuda(), batch['label'].cuda()

            optimizer.zero_grad()

            with amp.autocast(enabled=fp16):
                loss = model.forward_train(image=image, label=label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            lr_scheduler.step()

            train_mean_loss += loss.item()

            if step % log_interval == 0:
                s = step / step_per_epoch
                s = epoch + s
                logger.info(f'epoch: [{round(s, 1)}/{epochs}] - loss: {loss:.6f}')

        if valid_dataset is not None:
            model.eval()
            if tta:
                valid_loss, valid_score = valid_tta(model=model, valid_dataset=valid_dataset, threshold=threshold)
            else:
                valid_loss, valid_score = valid(model=model, valid_dataset=valid_dataset, threshold=threshold)
            logger.info(f'epoch: [{epoch + 1}/{epochs}] - valid loss: {valid_loss:.6f} - dice score: {valid_score:.6f}')
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_score = valid_score
                best_state = deepcopy(model.state_dict())

        else:
            if train_mean_loss < best_loss:
                best_loss = train_mean_loss
                best_score = 0.0
                best_state = deepcopy(model.state_dict())

    logger.info(f'Best loss: {best_loss:.6f}')
    logger.info(f'Best score: {best_score:.6f}')
    torch.save(best_state, f'{save_dir}/best_loss.pth')
    torch.save(model.state_dict(), f'{save_dir}/last.pth')

from tqdm import tqdm
import torch
from ..models.losses import accuracy, dice_score, iou_score


@torch.inference_mode()
def evalate(model, valid_loader, metrics: list = ['mdice'], fp16: bool = False, device: str = 'cpu'):
    _metric_dict = {'mdice': dice_score, 'miou': iou_score}
    metrics_dict = {name: _metric_dict[name] for name in metrics}
    metrics_dict['accuracy'] = accuracy

    scores = {metric_key: 0.0 for metric_key in metrics_dict.keys()}
    scores['n'] = 0

    for batch in tqdm(valid_loader):
        images, labels = batch['images'].to(device), batch['labels'].long().to(device)

        with torch.cuda.amp.autocast(enabled=fp16):
            predicts = model.forward_test(images)

        for metric_key, metric_fn in metrics_dict.items():
            score = metric_fn(predicts=predicts, labels=labels)
            scores[metric_key] += score
        
        scores['n'] += 1
    
    for metric_key in metrics_dict.keys():
        scores[metric_key] = scores[metric_key] / scores['n']
    _ = scores.pop('n')
    return scores
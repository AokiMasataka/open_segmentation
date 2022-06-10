from tqdm import tqdm
import torch
from torch.cuda import amp
from torch.nn import functional

from dataset import PostProcesser


@torch.no_grad()
def valid(model, valid_dataset, threshold=0.5):
    post_process = PostProcesser(threshold=threshold)
    model.eval()

    mean_loss = 0.0
    mean_score = 0.0

    num_datas = valid_dataset.__len__()
    for index in tqdm(range(num_datas)):
        batch = valid_dataset.__getitem__(index)

        batch['image'] = batch['image'].unsqueeze(0).cuda()
        batch['label'] = batch['label'].unsqueeze(0).cuda()

        with amp.autocast():
            pred = model.forward_test(image=batch['image'], label=batch['label'])
        loss, logit = pred['loss'].float(), pred['logit'].float()
        pred = post_process(results=batch, logit=logit)

        dice_score = valid_dataset.pre_eval(pred, index)

        mean_loss += loss.item()
        mean_score += dice_score

    mean_loss = mean_loss / num_datas
    mean_score = mean_score / num_datas
    return mean_loss, mean_score


@torch.no_grad()
def valid_tta(model, valid_dataset, threshold=0.5):
    post_process = PostProcesser(threshold=threshold)
    model.eval()

    mean_score = 0.0

    num_datas = valid_dataset.__len__()
    for index in tqdm(range(num_datas)):
        batch = valid_dataset.__getitem__(index)

        pred = test_time_augment(model=model, batch=batch, post_process=post_process, scales=(384, 288))

        dice_score = valid_dataset.pre_eval(pred, index)

        mean_score += dice_score

    mean_score = mean_score / num_datas
    return 0.0, mean_score


@torch.no_grad()
def tta(model, batch, post_process, scales=(384, 288)):
    image = batch['image']
    original_scale = image.shape[-1]
    preds = []

    image = torch.cat((image, torch.flip(image, dims=[2])), dim=0)
    with amp.autocast():
        logits = model(image)
    for logit in [logits[0], torch.flip(logits[0], dims=[2])]:
        preds.append(post_process(results=batch, logit=logit.float()))

    for scale in scales:
        rescale_image = functional.interpolate(image, size=scale, mode='bilinear')
        rescale_image = torch.cat((rescale_image, torch.flip(rescale_image, dims=[2])), dim=0)
        with amp.autocast():
            logits = model(rescale_image)
        logits = functional.interpolate(logits, size=original_scale, mode='bilinear')
        for logit in [logits[0], torch.flip(logits[0], dims=[2])]:
            preds.append(post_process(results=batch, logit=logit.float()))

    pred = sum(preds) / preds.__len__()
    return pred


@torch.no_grad()
def test_time_augment(model, batch, post_process, scales=(384, 288)):
    image = batch['image'].unsqueeze(0).cuda()
    preds = []

    for scale in (-1, *scales):
        if scale != -1:
            rescale_image = functional.interpolate(image, size=scale, mode='bilinear')
        else:
            rescale_image = image

        rescale_image = torch.cat((rescale_image, torch.flip(rescale_image, dims=[2])), dim=0)
        with amp.autocast():
            logits = model(rescale_image)

        for logit in [logits[0], torch.flip(logits[0], dims=[2])]:
            preds.append(post_process(results=batch, logit=logit.float()))

    pred = sum(preds) / preds.__len__()
    return pred


if __name__ == '__main__':
    x = torch.rand(1, 3, 320, 320)
    x = tta(x, flip=True, scales=(384, 288))

from tqdm import tqdm
import numpy
import torch


def evaluate_fn(model, dataset, threshold):
    mean_dice_score = 0.0
    progress_bar = tqdm(range(dataset.__len__()), desc='validating...')
    for index in progress_bar:
        dice_score = evaluate_one_step(model=model, dataset=dataset, index=index, threshold=threshold)
        mean_dice_score += dice_score

    mean_dice_score = mean_dice_score / dataset.__len__()
    return mean_dice_score


@torch.inference_mode()
def evaluate_one_step(model, dataset, index, threshold=0.5):
    batch = dataset.__getitem__(index)
    image = batch['image'].cuda()

    with torch.cuda.amp.autocast():
        logits = model.forward_inference(image)

    predict = dataset.pipeline.transforms['TestTimeAugment'].post_process(
        logits=logits, augmented_results=batch['meta']
    )

    predict = (threshold < predict).astype(numpy.float32)

    dice_score = dataset.pre_eval(pred=predict, index=index)
    return dice_score

from tqdm import tqdm
import numpy as np
import torch


@torch.no_grad()
def valid_fn(model, dataset, threshold=0.5):
    mean_dice_score = 0.0
    progress_bar = tqdm(range(dataset.__len__()), desc='validating...')
    for index in progress_bar:
        batch = dataset.__getitem__(index)
        images = batch['image'].cuda()
        augmented_results = batch['meta']

        with torch.cuda.amp.autocast():
            logits = model(images)

        predict = dataset.pipeline.transforms['TestTimeAugment'].post_process(
            logits=logits, augmented_results=augmented_results
        )

        predict = (threshold < predict).astype(np.float32)

        dice_score = dataset.pre_eval(pred=predict, index=index)
        mean_dice_score += dice_score

    mean_dice_score = mean_dice_score / dataset.__len__()
    return mean_dice_score

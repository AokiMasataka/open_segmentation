import torch

from utils import seed_everything
from dataset import PostProcesser
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
    return config


class InferenceSegmenter:
    def __init__(self, confg_file, weight_path, binarization=False):
        config = load_config_file(confg_file)
        seed_everything(config['seed'])
        self.model = builder.build_model(config)
        self.model.load_state_dict(torch.load(weight_path))
        self.model.cuda()
        self.model.eval()
        self.post_process = PostProcesser(threshold=config['threshold'], binarization=binarization)
        self.test_pipeline = builder.build_pipeline(config, mode='test_pipeline')

    @torch.inference_mode()
    def __call__(self, image):
        results = dict(image=image, original_shape=(image.shape[1], image.shape[0]))
        results = self.test_pipeline(results)
        logit = self.model(results['image'].unsqueeze(0).cuda())

        pred = self.post_process(results, logit)
        return pred


# from glob import glob
# import numpy as np
# import cv2
#
# scans_dir = 'D:/Dataset/uw-madison-gi-tract-image-segmentation/train/case123/case123_day20/scans/*.png'
#
# config = 'D:/Projects/open_segment/work_dir/eff/eff_fold0/config.py'
# weight = 'D:/Projects/open_segment/work_dir/eff/eff_fold0/best_loss.pth'
#
# segmenter = InferenceSegmenter(config, weight)
#
# index = 81
# image_scans = [path.replace('\\', '/') for path in glob(scans_dir)]
# image_path = image_scans[index]
#
# image = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)
# image = image.astype(np.float32)
# image = image / np.max(image)
# image = np.stack([image for _ in range(3)], axis=-1)
#
# pred = segmenter(image)

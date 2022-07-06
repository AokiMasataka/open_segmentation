import torch

from open_seg.utils import seed_everything
from open_seg.dataset.pipeline.post_process import PostProcesser
from open_seg import builder


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

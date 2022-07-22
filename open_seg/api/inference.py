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
        seed_everything(config['train_config']['seed'])
        if 'pretrained' in config['model']['backbone']:
            config['model']['backbone']['pretrained'] = False
        # self.model = builder.build_model(config['model'])
        self.model = builder.build_test_model(config=config['model'])
        self.model.load_state_dict(torch.load(weight_path))
        self.model.cuda()
        self.model.eval()
        self.post_process = PostProcesser(threshold=config['train_config']['threshold'], binarization=binarization)

        # test_pipeline = builder.build_pipeline(tta_config, mode='test_pipeline')
        # self.tta = test_pipeline.transforms.pop('TestTimeAugment')

        for pipe_config in config['test_pipeline']:
            if pipe_config['type'] == 'TestTimeAugment':
                tta_config = pipe_config
        self.tta = builder.PIPELINES.build(tta_config)

    @torch.inference_mode()
    def __call__(self, image):
        results = dict(image=image, original_shape=(image.shape[1], image.shape[0]))
        results = self.tta(results)
        # print(results['image'].shape)
        # exit()
        # logit = self.model(results['image'].unsqueeze(0).cuda())
        logit = self.model.forward_inference(results['image'].cuda())

        predict = self.tta.post_process(logits=logit, augmented_results=results['meta'])
        predict = (0.5 < predict).astype(int)
        # pred = self.post_process(results, predict)
        return predict

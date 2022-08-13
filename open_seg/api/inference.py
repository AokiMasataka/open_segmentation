import copy
import torch

from open_seg.utils import seed_everything
from open_seg import builder


def load_config_file(path):
    with open(path, 'r') as f:
        text = f.read()
    config = dict()
    exec(text, globals(), config)
    return config


class InferenceSegmenter:
    def __init__(self, confg_file, weight_path):
        config = load_config_file(confg_file)
        seed_everything(config['train_config']['seed'])
        if 'pretrained' in config['model']['backbone']:
            config['model']['backbone']['pretrained'] = False

        self.config = config
        self.model = builder.build_test_model(config=config['model'])
        if '.cpt' in weight_path:
            match_keys = self.model.load_state_dict(torch.load(weight_path)['model'])
        else:
            match_keys = self.model.load_state_dict(torch.load(weight_path))
        print(match_keys)
        self.model.cuda()
        self.model.eval()

        self.tta_config = copy.deepcopy(config['test_pipeline'][-1])
        self.tta = builder.PIPELINES.build(self.tta_config)
        self.pipeline = builder.build_pipeline(config=config, mode='test_pipeline')
        self.threshold = config['train_config'].get('threshold', 0.5)

    @torch.inference_mode()
    def __call__(self, image):
        results = dict(image=image, original_shape=(image.shape[1], image.shape[0]))
        results = self.tta(results)

        print('dtype: ', results['image'].dtype)
        print('min: ', results['image'].min().item(), ' - max: ', results['image'].max().item())
        logit = self.model.forward_inference(results['image'].cuda())

        predict = self.tta.post_process(logits=logit, augmented_results=results['meta'])
        predict = (0.5 < predict).astype(int)
        return predict

    @torch.inference_mode()
    def inference_filename(self, file_name):
        results = dict(image_path=file_name)
        results = self.pipeline(results)

        logits = self.model.forward_inference(results['image'].cuda())

        predict = self.pipeline.transforms['TestTimeAugment'].post_process(logits, results['meta'])
        # predict = (self.threshold < predict)
        return predict

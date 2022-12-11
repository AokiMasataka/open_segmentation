import numpy as np
import torch
from typing import Union

from ..utils import load_config_file
from ..models import build_model
from ..dataset import build_pipeline


class InferenceSegmentor:
    def __init__(self, config_path: str, weight_path: str):
        assert weight_path.endswith('.pth') or weight_path.endswith('.cpt')
        self.config_path = config_path
        self.weight_path = weight_path
        config, _ = load_config_file(config_path)
        self.model = build_model(config=config['model'])

        if weight_path.endswith('.pth'):
            miss_match_keys = self.model.load_state_dict(
                torch.load(weight_path, map_location='cpu'), strict=False
            )
        elif weight_path.endswith('.cpt'):
            miss_match_keys = self.model.load_state_dict(
                torch.load(weight_path, map_location='cpu')['model'], strict=False
            )

        print(miss_match_keys)

        self.pipeline = build_pipeline(config=config['data']['valid']['pipeline'])
        if 'LoadAnnotations' in self.pipeline.transforms:
            self.pipeline.transforms.pop('LoadAnnotations')

        self.threshold = config['train_config'].get('threshold', 0.5)

    @torch.inference_mode()
    def __call__(self, src: Union[str, np.ndarray]):
        if isinstance(src, str):
            results = {'image_path': src}
        elif isinstance(src, np.ndarray):
            results = {'image_path': None, 'image': src}
        else:
            TypeError(f'input type: {type(src)}')

        results = self.pipeline(results)

        with torch.cuda.amp.autocast():
            logits = self.model.forward_inference(results['image'])

        predict = self.pipeline.transforms['TestTimeAugment'].post_process(
            logits=logits, augmented_results=results['meta']
        )

        return (self.threshold < predict).astype(np.float32)

    def to(self, device):
        self.model.to(device)

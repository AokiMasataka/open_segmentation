import torch

from openback.utils import Registry
from openback import BACKBONES


DECODERS = Registry(name='decoders')
LOSSES = Registry(name='losses')
SEGMENTORS = Registry(name='segmentors')


def build_segmentor(config: dict, weight: str = None):
    assert isinstance(config, dict), f'config type: {type(config)}, must be dict type'
    segmentor = SEGMENTORS.build(config=config)
    if weight is not None:
        state_dict = torch.load(weight, map_location='cpu')
        missing_kys = segmentor.load_state_dict(state_dict, strict=False)
        print('miss match keys: ', missing_kys)
    return segmentor
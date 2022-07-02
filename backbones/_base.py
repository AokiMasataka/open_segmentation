import torch
from torch import nn


class BackboneBase(nn.Module):
    def __init__(self, init_config=None):
        super(BackboneBase, self).__init__()
        if init_config is not None:
            weight_path = init_config['weight_path']
            if weight_path is not None:
                print(f'load weight from {weight_path}')
                miss_match_key = self.load_state_dict(torch.load(weight_path), strict=False)
                print(miss_match_key)

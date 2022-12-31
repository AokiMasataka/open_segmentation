import torch
from torch import nn


class DecoderBase(nn.Module):
    def __init__(self, init_config=None):
        super(DecoderBase, self).__init__()
        assert isinstance(init_config, dict) or init_config is None
        self.init_config = init_config
    
    def init(self):
        if self.init_config is not None:
            if self.init_config.get('pretrained', False):
                state_dict = torch.load(self.init_config['pretrained'], map_location='cpu')

                for key in self.state_dict.keys():
                    if key in state_dict.keys():
                        if state_dict[key].shape == self.state_dict[key].shape:
                            self.state_dict[key] = state_dict[key]
            
            if self.init_config.get('', False):
                pass
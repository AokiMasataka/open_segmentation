import logging
import torch


class DecoderBase(torch.nn.Module):
    def __init__(self, init_config: dict = None):
        super(DecoderBase, self).__init__()
        self.init_config = init_config
    
    def init(self):
        if self.init_config is not None:
            if self.init_config.get('pretrained', False):
                state_dict = torch.load(self.init_config['pretrained'], map_location='cpu')
                match_keys = list()
                miss_match_keys = list()

                for key, value in state_dict.items():
                    miss_match = True
                    if key in self.state_dict().keys():
                        if self.state_dict()[key].shape == value.shape:
                            self.state_dict()[key] = value
                            miss_match = False
                            match_keys.append(key)
                    if miss_match:
                        miss_match_keys.append(key)
                
            logging.info(msg='Decoder match keys:')
            for match_key in match_keys:
                logging.info(msg=f'    {match_key}')
            
            logging.info(msg='Decoder miss match keys:')
            for miss_match_key in miss_match_keys:
                logging.info(msg=f'    {miss_match_key}')
    
    @property
    def num_classes(self):
        return self._num_classes
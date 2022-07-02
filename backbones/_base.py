from abc import ABCMeta, abstractmethod
from copy import deepcopy
from typing import Optional
import torch


class BackboneBase(torch.nn.Module, metaclass=ABCMeta):
    def __init__(self, init_config: Optional[dict] = None):
        super().__init__()
        self._is_init = False
        self.init_config = deepcopy(init_config)

        if self.init_config is not None:
            self.load_weight()
        else:
            self.init_weight()

    @property
    def is_init(self):
        return self._is_init

    def init_weight(self):
        self._is_init = True

    def load_weight(self):
        weight_path = self.init_config.get('weight_path', None)
        if weight_path is not None:
            state_dict = torch.load(weight_path, map_location='cpu')
            miss_match_key = self.load_state_dict(state_dict, strict=False)
            print(miss_match_key)

        self._is_init = True

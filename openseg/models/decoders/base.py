from openback.utils import BaseModule


class DecoderBase(BaseModule):
    def __init__(self, init_config: dict = None) -> None:
        super().__init__(init_config=init_config)
    
    @property
    def num_classes(self) -> None:
        return self._num_classes

import torch
from .base import SegmentorBase
from ..builder import SEGMENTORS, BACKBONES, DECODERS, LOSSES


@SEGMENTORS.register_module
class EncoderDecoder(SegmentorBase):
    def __init__(
        self,
        backbone: dict,
        decoder: dict,
        loss: dict,
        init_config: dict = None,
        norm_config: dict = None,
        test_config: dict = dict(mode='whole')
    ):
        super(EncoderDecoder, self).__init__(
            init_config=init_config,
            norm_config=norm_config,
            test_config=test_config
        )

        self.backbone = BACKBONES.build(config=backbone)
        self.decoder = DECODERS.build(config=decoder)
        self.losses = LOSSES.build(config=loss)
        self.init()
    
    def forward(self, images: torch.Tensor):
        """
        images: Tensor (batch, dim, w, h)
        """
        features = self.backbone(self.norm_fn(images=images))
        return self.decoder(features)
    
    def forward_train(self, images: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            images: Tensor (batch, dim, w, h)
            labels: Tensor (batch, w, h)
        """
        predicts = self(images=images)
        loss, losses = self._get_losses(predicts=predicts, labels=labels)
        return loss, losses
    
    @torch.inference_mode()
    def forward_test(self, images):
        if self.test_mode == 'whole':
            return self(images=images)
        else:
            return self.slide_inference(images=images)

    @property
    def num_classes(self):
        return self.decoder.num_classes
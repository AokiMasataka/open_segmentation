from torch import nn

from .base import SegmentorBase
from ..builder import SEGMENTER, BACKBONES, DECODERS, LOSSES
from openseg.utils.torch_utils import force_fp32


@SEGMENTER.register_module
class EncoderDecoder(SegmentorBase):
    def __init__(
            self,
            backbone: dict,
            decoder: dict,
            loss: dict,
            test_config=None,
            init_config=None,
            norm_config=None
    ):
        super(EncoderDecoder, self).__init__(
            num_classes=decoder['num_classes'],
            test_config=test_config,
            init_config=init_config,
            norm_config=norm_config
        )

        self.backbone = BACKBONES.build(config=backbone)
        self.decoder = DECODERS.build(config=decoder)
        self.losses = LOSSES.build(config=loss)

        self.init()

    def forward(self, images):
        images = self.norm_fn(images=images)
        logits = self.decoder(self.backbone(images))
        return logits

    def forward_train(self, images, labels):
        logits = self(images=images)
        loss, losses = self._get_loss(logits, labels)
        return loss, losses

    def forward_test(self, images, labels):
        logits = self(images=images)
        loss, _ = self._get_loss(logits, labels)
        return {'loss': loss, 'logit': logits}

    def forward_inference(self, images):
        if self.test_config['mode'] == 'whole':
            return self(images=images)
        elif self.test_config['mode'] == 'slide':
            return self.slide_inference(images=images)

    @force_fp32
    def _get_loss(self, logits, labels):
        losses = {}
        for loss_name, loss_fn in zip(self.losses.keys(), self.losses.values()):
            losses[loss_name] = loss_fn(logits, labels)
        loss = sum(losses.values())
        return loss, losses

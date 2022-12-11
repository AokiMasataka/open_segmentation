from torch import nn

from .base import SegmentorBase
from ..builder import SEGMENTER
from openseg.utils.torch_utils import force_fp32


@SEGMENTER.register_module
class EncoderDecoder(SegmentorBase):
    def __init__(
            self,
            backbone,
            decoder,
            losses,
            num_classes=1,
            test_config=None,
            init_config=None,
            norm_config=None
    ):
        super(EncoderDecoder, self).__init__(
            num_classes=num_classes,
            test_config=test_config,
            init_config=init_config,
            norm_config=norm_config
        )

        self.backbone = backbone
        self.decoder = decoder
        self.losses = losses

        self.seg_head = nn.Conv2d(
            in_channels=decoder.decoder_out_dim(),
            out_channels=num_classes,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1)
        )

        self.init()

    def forward(self, image):
        image = self.norm_fn(image=image)
        decode_out = self.decoder(self.backbone(image))
        return self.seg_head(decode_out)

    def forward_train(self, image, label):
        logit = self(image)
        loss, losses = self._get_loss(logit, label)
        return loss, losses

    def forward_test(self, image, label):
        logit = self(image)
        loss, _ = self._get_loss(logit, label)
        return {'loss': loss, 'logit': logit}

    def forward_inference(self, image):
        if self.test_config['mode'] == 'whole':
            return self(image=image)
        elif self.test_config['mode'] == 'slide':
            return self.slide_inference(image=image)

    @force_fp32
    def _get_loss(self, logit, label):
        losses = {}
        for loss_name, loss_fn in zip(self.losses.keys(), self.losses.values()):
            losses[loss_name] = loss_fn(logit, label)
        loss = sum(losses.values())
        return loss, losses


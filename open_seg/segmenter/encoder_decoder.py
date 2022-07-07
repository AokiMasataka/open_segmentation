from open_seg.segmenter._base import SegmenterBase, SegmentationHead
from open_seg.builder import SEGMENTER
from open_seg.utils import force_fp32


@SEGMENTER.register_module
class EncoderDecoder(SegmenterBase):
    def __init__(self, backbone, decoder, losses, num_classes=3, head_hidden_rate=1.0):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.losses = losses
        self.num_classes = num_classes

        self.seg_head = SegmentationHead(
            in_channels=self.decoder.decoder_out_dim, num_classes=num_classes, hidden_rate=head_hidden_rate
        )

    def forward(self, image):
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

    @force_fp32
    def _get_loss(self, logit, label):
        losses = {}
        for loss_name, loss_fn in zip(self.losses.keys(), self.losses.values()):
            losses[loss_name] = loss_fn(logit, label)
        loss = sum(losses.values())
        return loss, losses

from ._base import SegmenterBase, SegmentationHead
from open_seg.builder import SEGMENTER
from open_seg.utils import force_fp32
from open_seg.losses import lovasz_hinge_non_empty


@SEGMENTER.register_module
class EncoderDecoder(SegmenterBase):
    def __init__(self, backbone, decoder, losses, num_classes=1, head_hidden_rate=1.0, test_config=None, init_config=None):
        super(EncoderDecoder, self).__init__(num_classes=num_classes, test_config=test_config, init_config=init_config)
        self.backbone = backbone
        self.decoder = decoder
        self.losses = losses

        self.seg_head = SegmentationHead(
            in_channels=self.decoder.decoder_out_dim(),
            num_classes=num_classes,
            hidden_rate=head_hidden_rate
        )

    def forward(self, image):
        decode_out = self.decoder(self.backbone(image))
        return self.seg_head(decode_out)

    def forward_train(self, image, label):
        logit = self(image)
        loss, losses = self._get_loss(logit, label)
        return loss, losses

    def forward_test(self, image, label):
        decode_out = self.decoder(self.backbone(image))
        logit = self.seg_head(decode_out)
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


@SEGMENTER.register_module
class EncoderDecoderDeepVision(SegmenterBase):
    def __int__(self, backbone, decoder, losses, num_classes=1, head_hidden_rate=1.0, test_config=None, init_config=None):
        super(EncoderDecoderDeepVision, self).__int__(
            num_classes=num_classes,
            test_config=test_config,
            init_config=init_config
        )

        self.backbone = backbone
        self.decoder = decoder
        self.losses = losses

        self.seg_head = SegmentationHead(
            in_channels=self.decoder.decoder_out_dim(),
            num_classes=num_classes,
            hidden_rate=head_hidden_rate
        )

    def forward(self, image):
        decode_out = self.decoder.inference(self.backbone(image))
        return self.seg_head(decode_out)

    def forward_train(self, image, label):
        decode_out, deep_logits = self.decoder(self.backbone(image))
        logit = self.seg_head(decode_out)
        loss, losses = self._get_loss(logit, label)
        return loss, losses

    def forward_test(self, image, label):
        decode_out, deep_logits = self.decoder(self.backbone(image))
        logit = self.seg_head(decode_out)
        loss, _ = self._get_loss(logit, label)
        return {'loss': loss, 'logit': logit}

    def forward_inference(self, image):
        if self.test_config['mode'] == 'whole':
            return self(image=image)
        elif self.test_config['mode'] == 'slide':
            return self.slide_inference(image=image)

    @force_fp32
    def _get_loss(self, logit, deep_logits, label):
        losses = {}
        for loss_name, loss_fn in zip(self.losses.keys(), self.losses.values()):
            losses[loss_name] = loss_fn(logit, label)

        losses['deep_vision'] = self._deep_vivison_loss(deep_visions=deep_logits, label=label)
        loss = sum(losses.values())
        return loss, losses

    @force_fp32
    def _deep_vivison_loss(self, deep_visions, label):
        loss = sum([0.1 * lovasz_hinge_non_empty(logits_deep=logits_deep, label=label) for logits_deep in deep_visions])
        return loss

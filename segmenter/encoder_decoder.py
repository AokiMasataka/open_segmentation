from torch import nn

from segmenter._base import SegmenterBase
from builder import SEGMENTER
from utils import force_fp32


@SEGMENTER.register_module
class EncoderDecoder(SegmenterBase):
    def __init__(self, backbone, decoder, losses, num_classes=3):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.losses = losses
        self.num_classes = num_classes

        self.seg_head = SegmentationHead(self.decoder.decoder_out_dim, num_classes=num_classes)

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


class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes, kernel_size=3, upsampling=1):
        super(SegmentationHead, self).__init__()
        kernel_size_t2 = (kernel_size, kernel_size)
        module_list = []
        if 1 < upsampling:
            module_list.append(nn.UpsamplingBilinear2d(scale_factor=upsampling))
        module_list.append(
            nn.Conv2d(in_channels, num_classes, kernel_size=kernel_size_t2, padding=kernel_size // 2)
        )

        self.body = nn.Sequential(*module_list)

    def forward(self, x):
        return self.body(x)

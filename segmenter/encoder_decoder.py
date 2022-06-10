from torch import nn

from builder import SEGMENTER


@SEGMENTER.register_module
class EncoderDecoder(nn.Module):
    def __init__(self, backbone, decoder, losses, num_classes=3):
        super(EncoderDecoder, self).__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.losses = losses
        self.num_classes = num_classes

        self.seg_head = SegmentationHead(self.decoder._decoder_channels[-1], num_classes=num_classes)

    def _encode_decode(self, image):
        features = self.backbone(image)
        return self.decoder(*features)

    def forward(self, image):
        feature = self._encode_decode(image)
        return self.seg_head(feature)

    def forward_train(self, image, label):
        logit = self(image)
        loss = self._get_loss(logit, label)
        return loss

    def forward_test(self, image, label):
        logit = self(image)
        loss = self._get_loss(logit, label)
        return {'loss': loss, 'logit': logit}

    def _get_loss(self, logit, label):
        losses = []
        for loss_name, loss_fn in zip(self.losses.keys(), self.losses.values()):
            losses.append(loss_fn(logit, label))
        loss = sum(losses)
        return loss


class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes, kernel_size=3, upsampling=1):
        super(SegmentationHead, self).__init__()
        kernel_size_t2 = (kernel_size, kernel_size)
        module_list = [
            nn.Conv2d(in_channels, num_classes, kernel_size=kernel_size_t2, padding=kernel_size // 2)]
        if 1 < upsampling:
            module_list.append(nn.UpsamplingBilinear2d(scale_factor=upsampling))

        self.body = nn.Sequential(*module_list)

    def forward(self, x):
        return self.body(x)
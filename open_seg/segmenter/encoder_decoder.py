import torch
from torch.nn import functional
from ._base import SegmenterBase, SegmentationHead
from open_seg.builder import SEGMENTER
from open_seg.utils import force_fp32


@SEGMENTER.register_module
class EncoderDecoder(SegmenterBase):
    def __init__(self, backbone, decoder, losses, num_classes=1, head_hidden_rate=1.0, test_config=None):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.losses = losses
        self.num_classes = num_classes
        self.test_config = test_config

        self.seg_head = SegmentationHead(
            in_channels=self.decoder.decoder_out_dim(), num_classes=num_classes, hidden_rate=head_hidden_rate
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

    @torch.inference_mode()
    def slide_inference(self, image):
        h_stride, w_stride = self.test_config['stride']
        h_crop, w_crop = self.test_config['crop_size']
        num_classes = self.num_classes

        batch, _, h_img, w_img = image.size()
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = image.new_zeros((batch, num_classes, h_img, w_img))
        count_mat = image.new_zeros((batch, 1, h_img, w_img))
        crop_image_list = []
        pad_meta_list = []

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                count_mat[:, :, y1:y2, x1:x2] += 1
                crop_image_list.append(image[:, :, y1:y2, x1:x2])
                pad_meta_list.append((int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2)))

        crop_image = torch.cat(crop_image_list, dim=0)
        crop_logit_list = [image for image in self(crop_image)]

        for crop_logit, pad_meta in zip(crop_logit_list, pad_meta_list):
            preds += functional.pad(input=crop_logit, pad=pad_meta, mode='constant', value=0)

        assert (count_mat == 0).sum() == 0
        preds = preds / count_mat
        return preds

    @torch.inference_mode()
    def slide_inference_one_image(self, image):
        h_stride, w_stride = self.test_config['stride']
        h_crop, w_crop = self.test_config['crop_size']
        _, h_img, w_img = image.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = image.new_zeros((num_classes, h_img, w_img))
        count_mat = image.new_zeros((1, h_img, w_img))
        crop_image_list = []
        pad_meta_list = []
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                count_mat[:, y1:y2, x1:x2] += 1
                crop_image_list.append(image[:, y1:y2, x1:x2])
                pad_meta_list.append((int(x1), int(preds.shape[2] - x2), int(y1), int(preds.shape[1] - y2)))

        crop_image = torch.stack(crop_image_list, dim=0)
        crop_logit_list = [image for image in self(crop_image)]

        for pad_meta, crop_logit in zip(pad_meta_list, crop_logit_list):
            preds += functional.pad(input=crop_logit, pad=pad_meta, mode='constant', value=0)

        assert (count_mat == 0).sum() == 0
        preds = preds / count_mat
        return preds

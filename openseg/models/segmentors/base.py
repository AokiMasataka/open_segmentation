import logging
from copy import deepcopy
from abc import ABCMeta, abstractmethod

import torch
from torch import nn
from torch.nn import functional
from openseg.utils.torch_utils import force_fp32


class SegmentorBase(torch.nn.Module, metaclass=ABCMeta):
    def __init__(self, num_classes, test_config=None, norm_config=None, init_config=None):
        super().__init__()
        self._is_init = False
        self.test_config = deepcopy(test_config)
        self.norm_config = deepcopy(norm_config)
        self.init_config = deepcopy(init_config)

        if self.norm_config is not None:
            assert self.norm_config['mean'].__len__() == self.norm_config['std'].__len__()
            ch = self.norm_config['mean'].__len__()
            mean = torch.tensor(self.norm_config['mean'], dtype=torch.float).view(1, ch, 1, 1)
            std = torch.tensor(self.norm_config['std'], dtype=torch.float).view(1, ch, 1, 1)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
            self.norm_fn = self._norm
        else:
            self.mean = None
            self.std = None
            self.norm_fn = nn.Identity()

        self.num_classes = num_classes

    def _norm(self, images):
        return (images.float() / 255.0 - self.mean) / self.std
    
    def init(self):
        if self.init_config.get('pretrained', False):
            state_dict = torch.load(self.init_config['pretrained'], map_location='cpu')
            if 'model' in state_dict.keys():
                state_dict = state_dict['model']

            match_keys = []
            miss_match_keys = []
            for key, value in state_dict.items():
                miss_match = True
                if key in self.state_dict().keys():
                    if self.state_dict()[key].shape == value.shape:
                        self.state_dict()[key] = value
                        miss_match = False
                        match_keys.append(key)
                if miss_match:
                    miss_match_keys.append(key)
            
            logging.info(msg='segmentor: match keys:')
            if self.init_config.get('print_match_key', False):
                print('segmentor: match keys:')
            for match_key in match_keys:
                logging.info(msg=f'    {match_key}')
                if self.init_config.get('print_match_key', False):
                    print('    ', match_key)
                
            logging.info(msg='segmentor: miss match keys:')
            if self.init_config.get('print_miss_match_key', False):
                print('segmentor: miss match keys:')
            for miss_match_key in miss_match_keys:
                logging.info(msg=f'    {miss_match_key}')
                if self.init_config.get('print_miss_match_key', False):
                    print('    ', miss_match_key)

            logging.info(msg=f'segmentor: number of match keys: {match_keys.__len__()}')
            logging.info(msg=f'segmentor: number of miss match keys: {miss_match_keys.__len__()}')
            print(f'segmentor: number of match keys: {match_keys.__len__()}')
            print(f'segmentor: number of miss match keys: {miss_match_keys.__len__()}')

        if self.test_config is None:
            self.test_config = dict(mode='whole')

    @abstractmethod
    def forward_train(self, images, labels):
        pass

    @abstractmethod
    def forward_test(self, images, labels):
        pass

    @force_fp32
    def _get_loss(self, logits, labels):
        losses = dict()
        for loss_name, loss_fn in self.losses.items():
            losses[loss_name] = loss_fn(logits, labels)
        loss = sum(losses.values())
        return loss, losses

    @torch.inference_mode()
    def slide_inference(self, images):
        h_stride, w_stride = self.test_config['stride']
        h_crop, w_crop = self.test_config['crop_size']
        num_classes = self.num_classes

        batch, _, h_img, w_img = images.size()
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = images.new_zeros((batch, num_classes, h_img, w_img))
        count_mat = images.new_zeros((batch, 1, h_img, w_img))
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
                crop_image_list.append(images[:, :, y1:y2, x1:x2])
                pad_meta_list.append((int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2)))

        crop_image = torch.cat(crop_image_list, dim=0)
        crop_logit_list = [image for image in self(crop_image)]

        for crop_logit, pad_meta in zip(crop_logit_list, pad_meta_list):
            preds += functional.pad(input=crop_logit, pad=pad_meta, mode='constant', value=0)

        assert (count_mat == 0).sum() == 0
        preds = preds / count_mat
        return preds

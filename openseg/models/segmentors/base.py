import logging
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional


class SegmentorBase(nn.Module):
    def __init__(self, init_config: dict = None, norm_config: dict = None, test_config: dict = dict(mode='whole')):
        super(SegmentorBase, self).__init__()
        self.init_config = deepcopy(init_config)
        self.norm_config = deepcopy(norm_config)
        self.test_config = deepcopy(test_config)

        if self.norm_config is not None:
            assert self.norm_config['mean'].__len__() == self.norm_config['std'].__len__()
            ch = self.norm_config['mean'].__len__()
            mean = torch.tensor(self.norm_config['mean'], dtype=torch.float).view(1, ch, 1, 1)
            std = torch.tensor(self.norm_config['std'], dtype=torch.float).view(1, ch, 1, 1)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
            self.norm_fn = self._norm
            self.div = norm_config.get('div', None)
        else:
            self.mean = None
            self.std = None
            self.div = None
            self.norm_fn = nn.Identity()
        
        if test_config is not None:
            self.test_mode = test_config.get('mode', 'whole')
        else:
            self.test_mode = 'whole'

    def _norm(self, images):
        if self.div is None:
            return (images - self.mean) / self.std
        else:
            return (images / self.div - self.mean) / self.std
    
    def _get_losses(self, predicts, labels):
        predicts = predicts.float()
        losses = dict()
        for loss_name, loss_func in self.losses.items():
            losses[loss_name] = loss_func(predicts=predicts, labels=labels)
        loss = sum(losses.values())
        return loss, losses
    
    def init(self):
        if self.init_config is not None:
            if self.init_config.get('pretrained', False):
                state_dict = torch.load(self.init_config['pretrained'], map_location='cpu')
                match_keys = list()
                miss_match_keys = list()

                for key, value in state_dict.items():
                    miss_match = True
                    if key in self.state_dict().keys():
                        if self.state_dict()[key].shape == value.shape:
                            self.state_dict()[key] = value
                            miss_match = False
                            match_keys.append(key)
                    if miss_match:
                        miss_match_keys.append(key)
            
            if self.init_config.get('log_kys', False):
                print('segmentor match keys:')
                for match_key in match_keys:
                    print(f'    {match_key}')
                
                print('segmentor miss match keys:')
                for miss_match_key in miss_match_keys:
                    print(f'    {miss_match_key}')
                print(f'segmentor match keys: {match_keys.__len__()}')
                print(f'segmentor miss match keys: {miss_match_keys.__len__()}')
            
            logging.info(msg=f'segmentor match keys: {match_keys.__len__()}')
            logging.info(msg=f'segmentor miss match keys: {miss_match_keys.__len__()}')
            

    @torch.inference_mode()
    def slide_inference(self, images):
        h_stride, w_stride = self.test_config['stride']
        h_crop, w_crop = self.test_config['crop_size']

        batch, _, h_img, w_img = images.size()
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = images.new_zeros((batch, self.num_classes, h_img, w_img))
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
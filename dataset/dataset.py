from copy import deepcopy
import torch
from dataset.pipeline import LoadAnnotations
from losses import dice_metric


class SegmentData(torch.utils.data.Dataset):
    def __init__(self, split, data_root, image_dir, label_dir, suffix, test_mode=False):
        self.split_file = data_root + '/' + split
        self.image_dir = data_root + '/' + image_dir
        self.laebl_dir = data_root + '/' + label_dir
        self.suffix = suffix
        self.test_mode = test_mode
        self.pipeline = None
        self.gt_seg_map_loader = LoadAnnotations()

        self._load_annotation()

    def __len__(self):
        return self.ant_data_list.__len__()

    def __getitem__(self, index):
        return self._prepare_train_image(index)

    def _prepare_train_image(self, index):
        item = deepcopy(self.ant_data_list[index])
        return self.pipeline(item)

    def _load_annotation(self):
        self.ant_data_list = []
        with open(self.split_file, 'r') as f:
            splits = f.read().split('\n')

        for split in splits:
            if split == '':
                continue
            ant_data = {
                'image_path': f'{self.image_dir}/{split}{self.suffix}',
                'label_path': f'{self.laebl_dir}/{split}{self.suffix}'
            }
            self.ant_data_list.append(ant_data)

    def get_pipeline(self, pipeline):
        self.pipeline = pipeline

    def pre_eval(self, pred, index):
        assert pred.ndim == 3
        item = self.ant_data_list[index]
        results = self.gt_seg_map_loader(item)
        label = results['label']
        dice_score = dice_metric(pred, label, eps=1e-6)
        return dice_score


def train_collate_fn(batch):
    new_batch = {'image': [], 'label': [], 'original_shape': [], 'pad_t': [], 'pad_b': [], 'pad_l': [], 'pad_r': []}
    for one_batch in batch:
        new_batch['image'].append(one_batch['image'])
        new_batch['label'].append(one_batch['label'])

        new_batch['original_shape'].append(one_batch['original_shape'])
        new_batch['pad_t'].append(one_batch['pad_t'])
        new_batch['pad_b'].append(one_batch['pad_b'])
        new_batch['pad_l'].append(one_batch['pad_l'])
        new_batch['pad_r'].append(one_batch['pad_r'])

    new_batch['image'] = torch.stack(new_batch['image'], dim=0)
    new_batch['label'] = torch.stack(new_batch['label'], dim=0)
    return new_batch

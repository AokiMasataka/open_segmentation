from copy import deepcopy
import torch
from dataset.pipeline.loading import LoadAnnotations
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


def train_collate_fn(batch_list):
    images = []
    labels = []
    for batch in batch_list:
        images.append(batch.pop('image'))
        labels.append(batch.pop('label'))
    images = torch.stack(images, dim=0)
    labels = torch.stack(labels, dim=0)
    return {'image': images, 'label': labels}


def test_collate_fn(batch_list):
    images = []
    for batch in batch_list:
        images.append(batch.pop('image'))
    images = torch.cat(images, dim=0)
    return {'images': images, 'metas': batch_list}

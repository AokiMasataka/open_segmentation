from copy import deepcopy
import torch
from open_seg.losses import dice_metric


class SegmentData(torch.utils.data.Dataset):
    def __init__(
            self,
            split,
            data_root,
            image_prefix,
            label_prefix,
            image_suffix='.png',
            label_suffix='.png',
            cache_image=False,
            cache_label=False,
            test_mode=False
    ):
        self.split_file = data_root + '/' + split
        self.image_dir = data_root + '/' + image_prefix
        self.laebl_dir = data_root + '/' + label_prefix
        self.image_suffix = image_suffix
        self.label_suffix = label_suffix
        self.test_mode = test_mode
        self.pipeline = None
        self.gt_seg_map_loader = None

        self._load_annotation()

        self.cache_image = cache_image
        self.cache_label = cache_label

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
                'image_path': f'{self.image_dir}/{split}{self.image_suffix}',
                'label_path': f'{self.laebl_dir}/{split}{self.label_suffix}'
            }
            self.ant_data_list.append(ant_data)

    def _cache_images(self):
        for index in range(self.ant_data_list.__len__()):
            self.ant_data_list[index] = self.pipeline['LoadImageFromFile'](self.ant_data_list[index])

        if 'LoadImageFromFile' in self.pipeline.transforms.keys():
            del self.pipeline.transforms['LoadImageFromFile']

    def _cache_labels(self):
        for index in range(self.ant_data_list.__len__()):
            self.ant_data_list[index] = self.gt_seg_map_loader(self.ant_data_list[index])

        if 'LoadAnnotations' in self.pipeline.transforms.keys():
            del self.pipeline.transforms['LoadAnnotations']

    def get_pipeline(self, pipeline):
        self.pipeline = pipeline
        self.gt_seg_map_loader = pipeline.transforms['LoadAnnotations']

        if self.cache_image:
            self._cache_images()

        if self.cache_label:
            self._cache_labels()

    def pre_eval(self, pred, index):
        assert pred.ndim == 3
        item = self.ant_data_list[index]
        results = self.gt_seg_map_loader(item)
        label = results['label']
        dice_score = dice_metric(pred=pred, label=label, smooth=1.0)
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

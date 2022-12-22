import os
import itertools
from pathlib import Path
from copy import deepcopy
import torch
from torch.utils.data import Dataset
from .builder import build_pipeline
from ..models.losses import dice_metric


class CustomDataset(Dataset):
    def __init__(
            self,
            pipeline,
            data_root,
            split,
            image_prefix,
            label_prefix,
            image_suffix,
            label_suffix,
            cache_image=False,
            cache_label=False,
    ):
        self.split_file = os.path.join(data_root, split) if split is not None else None
        self.image_prefix = os.path.join(data_root, image_prefix)
        self.label_prefix = os.path.join(data_root, label_prefix)
        self.image_suffix = image_suffix
        self.label_suffix = label_suffix
        self.pipeline = None
        self.gt_seg_map_loader = None

        self._load_annotation()
        self.build_pipeline(pipeline_config=pipeline)

        if cache_image:
            self._cache_images()

        if cache_label:
            self._cache_labels()

    def __len__(self):
        return self.ant_data_list.__len__()

    def __getitem__(self, index):
        item = deepcopy(self.ant_data_list[index])
        return self.pipeline(item)

    def _load_annotation(self):
        self.ant_data_list = []
        if self.split_file is None:
            file_list = os.listdir(path=self.image_prefix)

            for file in file_list:
                file = Path(file).stem
                ant_data = {
                    'image_path': os.path.join(self.image_prefix, file + self.image_suffix),
                    'label_path': os.path.join(self.label_prefix, file + self.label_suffix)
                }
                self.ant_data_list.append(ant_data)

        else:
            with open(self.split_file, 'r') as f:
                splits = f.read().split('\n')

            for split in splits:
                if split == '':
                    continue
                ant_data = {
                    'image_path': os.path.join(self.image_prefix, split + self.image_suffix),
                    'label_path': os.path.join(self.label_prefix, split + self.label_suffix)
                }
                self.ant_data_list.append(ant_data)

    def _cache_images(self):
        # TODO parallel cache
        for index in range(self.ant_data_list.__len__()):
            self.ant_data_list[index] = self.pipeline['LoadImageFromFile'](self.ant_data_list[index])

        if 'LoadImageFromFile' in self.pipeline.transforms.keys():
            del self.pipeline.transforms['LoadImageFromFile']

    def _cache_labels(self):
        # TODO parallel cache
        for index in range(self.ant_data_list.__len__()):
            self.ant_data_list[index] = self.gt_seg_map_loader(self.ant_data_list[index])

        if 'LoadAnnotations' in self.pipeline.transforms.keys():
            del self.pipeline.transforms['LoadAnnotations']

    def pre_eval(self, pred, index):
        assert pred.ndim == 3, f'input shape is: {pred.shape}'
        item = self.ant_data_list[index]
        results = self.gt_seg_map_loader(item)
        label = results['label']
        dice_score = dice_metric(pred=pred, label=label, smooth=1.0)
        return dice_score

    def build_pipeline(self, pipeline_config):
        self.pipeline = build_pipeline(config=pipeline_config)
        self.gt_seg_map_loader = self.pipeline.transforms['LoadAnnotations']


class InfiniteSampler:
    def __init__(self, dataset_size):
        self.dataset_size = dataset_size
        self.generator = torch.Generator(device='cpu')

    def __iter__(self):
        yield from itertools.islice(self._infinite(), 0, None, 1)

    def _infinite(self):
        while True:
            yield from torch.randperm(self.dataset_size, generator=self.generator)


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

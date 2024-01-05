import os
import itertools
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from multiprocessing.pool import Pool
import torch
from torch.utils.data import Dataset
from .builder import DATASETS, build_pipeline


@DATASETS.register_module
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
        self.split_file =  None if split is None else os.path.join(data_root, split)
        self.image_prefix = os.path.join(data_root, image_prefix)
        self.label_prefix = os.path.join(data_root, label_prefix)
        self.image_suffix = image_suffix
        self.label_suffix = label_suffix
        self.pipeline = None
        self.gt_seg_map_loader = None

        self.ant_data_list = self._load_annotation()
        self.pipeline = build_pipeline(config=pipeline)

        if cache_image:
            self.ant_data_list = self._cache_images()

        if cache_label:
            self.ant_data_list = self._cache_labels()

    def __len__(self):
        return self.ant_data_list.__len__()

    def __getitem__(self, index):
        item = deepcopy(self.ant_data_list[index])
        return self.pipeline(item)

    def _load_annotation(self):
        ant_data_list = list()
        if self.split_file is None:
            file_names = [Path(file_name).stem for file_name in os.listdir(self.image_prefix)]
            for file_name in file_names:
                ant_data = {
                    'image_path': os.path.join(self.image_prefix, file_name + self.image_suffix),
                    'label_path': os.path.join(self.label_prefix, file_name + self.label_suffix)
                }
                ant_data_list.append(ant_data)
            return ant_data_list
        else:
            with open(self.split_file, 'r') as f:
                splits = f.read().strip().split('\n')
            for split in splits:
                if split == '':
                    continue
                ant_data = {
                    'image_path': os.path.join(self.image_prefix, split + self.image_suffix),
                    'label_path': os.path.join(self.label_prefix, split + self.label_suffix)
                }
                ant_data_list.append(ant_data)
            return ant_data_list

    def _cache_images(self):
        image_load_fn = self.pipeline.transforms.get('LoadImageFromFile', None)

        iterable = zip(self.ant_data_list, itertools.repeat(image_load_fn))
        with Pool(os.cpu_count() - 4) as pool:
            data = pool.imap_unordered(func=CustomDataset._cache, iterable=iterable)
            ant_data_list = [a for a in data]
        
        return ant_data_list

    def _cache_labels(self):
        image_load_fn = self.pipeline.transforms.get('LoadAnnotations', None)

        iterable = zip(self.ant_data_list, itertools.repeat(image_load_fn))
        with Pool(os.cpu_count() - 4) as pool:
            data = pool.imap_unordered(func=CustomDataset._cache, iterable=iterable)
            ant_data_list = tqdm([a for a in data])
        
        return ant_data_list

    @staticmethod
    def _cache(args):
        result, load_fn = args
        return load_fn(result=result)
    
    @staticmethod
    def train_collate_fn(batch_list):
        images = list()
        labels = list()
        for batch in batch_list:
            images.append(batch.pop('image'))
            labels.append(batch.pop('label'))
        images = torch.stack(images, dim=0)
        labels = torch.stack(labels, dim=0)
        return {'images': images, 'labels': labels}

    @staticmethod
    def valid_collate_fn(batch_list):
        images = list()
        labels = list()
        for batch in batch_list:
            images.append(batch.pop('image'))
            labels.append(batch.pop('label'))
        images = torch.stack(images, dim=0)
        labels = torch.stack(labels, dim=0)
        return {'images': images, 'labels': labels}
    
    @staticmethod
    def test_collate_fn(batch_list):
        images = list()
        for batch in batch_list:
            images.append(batch.pop('image'))
        images = torch.cat(images, dim=0)
        return {'images': images, 'metas': batch_list}


class InfiniteSampler:
    def __init__(self, dataset_size):
        self.dataset_size = dataset_size
        self.generator = torch.Generator(device='cpu')

    def __iter__(self):
        yield from itertools.islice(self._infinite(), 0, None, 1)

    def _infinite(self):
        while True:
            yield from torch.randperm(self.dataset_size, generator=self.generator)

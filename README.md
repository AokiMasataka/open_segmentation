# Introduction
open_segmentation is a library that allows anyone to easily perform segmentation from learning to inference.


# Installation
```bash
git clone https://github.com/AokiMasataka/open_segmentation.git
cd ./open_segmentation
pip install -r requirements.txt
python setup.py install
```

# config file

The config file is written in python, see `./configs/` for a template.

## model config

Basically, it is written as follows

```python
model = dict(
    segmenter=dict(
        type='EncoderDecoder',
    ),
    backbone=dict(
        type='resnet50',
        pretrained=True,
        n_blocks=5,
    ),
    decoder=dict(
        type='Unet',
        decoder_channels=(256, 128, 64, 32, 16),
        n_blocks=5,
    ),
    loss=[dict(type='CrossEntropyLoss', mode='bce', label_smooth=0.01, loss_weight=1.0)],
    init_config=None,
    test_config=dict(mode='whole'),
    norm_config=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    num_classes=10
)
```

Loss functions can be added as needed as follows

```python
loss=[
    dict(type='CrossEntropyLoss', mode='bce', label_smooth=0.01, loss_weight=1.0),
    dict(type='DiceLoss', mode='multiclass', smooth=1.0, loss_weight=1.0)    
],
```

## data pipeline
The testpipeline is used during verification.
TestTimeAugment can be inferred at multiple scales and flips. In the template, it is single scale, but
The template is for a single scale, but you can infer multiple scales by writing
`scales=(256, 256, 224, 224), flips=(False, True, False, True)`.

```python
train_pipeline = [
    dict(type='LoadImageFromFile', to_float=True, max_value='max', backend='cv2'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', size=256, keep_retio=True),
    dict(type='RandomFlipHorizontal', prob=0.5),
    dict(type='Padding', pad_value=0, label_pad_value=255),
]


test_pipeline = [
    dict(type='LoadImageFromFile', to_float=True, max_value='max', backend='cv2'),
    dict(type='LoadAnnotations'),
    dict(type='TestTimeAugment', input_size=256, scales=(256), flips=(False), keep_retio=True)
]
```

You can use Album if you have Album installed. Write as follows

```python
album_transform = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=0,
        interpolation=1,
        p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2),
    dict(type='ChannelShuffle', p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.1),
]

train_pipeline = [
    dict(type='LoadImageFromFile', to_float=True, max_value='max', backend='cv2'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', size=256, keep_retio=True),
    dict(type='Album', transforms=album_transform),
    dict(type='RandomFlipHorizontal', prob=0.5),
    dict(type='Padding', pad_value=0, label_pad_value=255),
]
```

# Data format

```python
data = dict(
    batch_size=32,
    num_workers=8,
    train=dict(
        pipeline=train_pipeline,
        data_root='dataset/path',
        split='train/train_split.txt',
        image_prefix='image_dir',
        label_prefix='label_dir',
        image_suffix='.png',
        label_suffix='.png',
        cache_image=False,
        cache_label=False,
    ),
    valid=dict(
        pipeline=test_pipeline,
        data_root='dataset/path',
        split='valid/valid_split.txt',
        image_prefix='image_dir',
        label_prefix='label_dir',
        image_suffix='.png',
        label_suffix='.png',
        cache_image=False,
        cache_label=False
    )
)
```

A training pair will consist of the files with same suffix in image_dir/label_dir.

described in `split` file are read, only part of the files in image_dir/label_dir will be loaded. We may specify the prefix of files we would like to be included in the split txt.

More specifically, for a split txt like following,

```txt
aaa
bbb
ccc
```

image:<br>
`{data_root}/{image_prefix}/aaa{image_suffix}`<br>
`{data_root}/{image_prefix}/bbb{image_suffix}`<br>
`{data_root}/{image_prefix}/ccc{image_suffix}`<br>
label:<br>
`{data_root}/{label_prefix}/aaa{label_suffix}`<br>
`{data_root}/{label_prefix}/bbb{label_suffix}`<br>
`{data_root}/{label_prefix}/ccc{label_suffix}`<br>
The above data is loaded.



# run trainning
```sh
python ./tools/launcher.py --config {config}.py
```

# inference

segmentor can be populated with both arrays and image paths.
In weight_path, enter the path of the weight to be trained. The path extension can be either `.pth` or `.cpt`.

```python
# input image array
import cv2
from open_seg import InferenceSegmentor

segmentor = InferenceSegmentor(config_path='{config}.py', weight_path='{weight}.pth')
image = cv2.imread('image_file.png')

predict = segmentor(src=image)
cv2.imshow('predict', predict)
cv2.waitKey(0)
```

```python
# input image path
import cv2
from open_seg import InferenceSegmentor

segmentor = InferenceSegmentor(config_path='{config}.py', weight_path='{weight}.pth')

predict = segmentor(src='image_file.png')
cv2.imshow('predict', predict)
cv2.waitKey(0)
```
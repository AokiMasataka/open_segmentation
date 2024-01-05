image_size = 384
train_pipeline = [
    dict(type='LoadImageFromFile', to_float=False, max_value=None, backend='cv2'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', size=image_size, keep_retio=False),
    dict(type='RandomFlipHorizontal', prob=0.5),
    dict(type='ShiftScaleRotateShear', rotate_limit=90, scale_limit=0.15, shift_limit=0.1, shear=0.01, label_pad_value=255),
    dict(type='ToTensor')
]


test_pipeline = [
    dict(type='LoadImageFromFile', to_float=False, max_value=None, backend='cv2'),
    dict(type='LoadAnnotations'),
    dict(
        type='TestTimeAugment',
        input_size=image_size,
        process_configs=[
            {'size': image_size, 'flip': False, 'keep_retio': False},
            {'size': image_size, 'flip': True, 'keep_retio': False},
        ],
        pad_value=0,
        label_pad_value=255
    )
]

data = dict(
	train=dict(
		batch_size=16,
		num_workers=8,
		dataset=dict(
			type='CustomDataset',
			pipeline=train_pipeline,
            data_root='dataset/path',
            split='train/train_split.txt',
            image_prefix='image_dir',
            label_prefix='label_dir',
            image_suffix='.jpg',
            label_suffix='.png',
            cache_image=False,
            cache_label=False,
        ),
    ),
	valid=dict(
		batch_size=16,
		num_workers=8,
		dataset=dict(
			type='CustomDataset',
			pipeline=test_pipeline,
            data_root='dataset/path',
            split='valid/valid_split.txt',
            image_prefix='image_dir',
            label_prefix='label_dir',
            image_suffix='.jpg',
            label_suffix='.png',
            cache_image=False,
            cache_label=False
        )
    ),
)

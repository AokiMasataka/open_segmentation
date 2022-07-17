model = dict(
    segmenter=dict(
        type='EncoderDecoder',
        num_classes=10,
    ),
    backbone=dict(
        type='resnet50',
        pretrained=True,
        n_blocks=5,
        init_config=dict(weight_path=None)
    ),
    decoder=dict(
        type='Unet',
        decoder_channels=(256, 128, 64, 32, 16),
        n_blocks=5,
        block_type='basic'
    ),
    loss=[dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.75)]
)

total_step = 60_000
optimizer = dict(type='AdamW', lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-6)
scheduler = dict(type='CosineAnnealingLR', T_max=total_step, eta_min=5e-7)


image_size = 384
album_transforms = [
    dict(type='RandomBrightnessContrast', p=0.5),
    dict(type='ShiftScaleRotate', shift_limit=0.2, scale_limit=0.075, rotate_limit=20, p=0.75),
    dict(
        type='OneOf',
        transforms=[
            dict(type='ElasticTransform', alpha=3, sigma=75, alpha_affine=75, p=1.0),
            dict(type='GridDistortion', num_steps=5, distort_limit=0.075, p=1.0)
        ],
        p=0.75
    ),
]


train_pipeline = [
    dict(type='LoadImageFromFile', to_float=True, color_type='anydepth', max_value='max', force_3chan=False),
    dict(type='LoadAnnotations'),
    dict(type='Resize', size=image_size, keep_retio=True),
    dict(type='RandomFlipHorizontal', prob=0.5),
    dict(type='Album', transforms=album_transforms),
    dict(type='Padding', pad_value=0, label_pad_value=255),
]


test_pipeline=[
    dict(type='LoadImageFromFile', to_float=True, color_type='anydepth', max_value='max', force_3chan=False),
    dict(type='LoadAnnotations'),
    dict(type='TestTimeAugment', scales=(320, 320, 288, 288), flips=(False, True, False, True), size=image_size)
]


data_root = 'dataset/path'
data = dict(
    batch_size=32,
    num_workers=8,
    train=dict(
        data_root=data_root,
        split='train/split',
        image_dir='image_dir',
        label_dir='label_dir',
        suffix='.png'
    ),
    valid=dict(
        data_root=data_root,
        split='valid/split',
        image_dir='image_dir',
        label_dir='label_dir',
        suffix='.png'
    )
)


train_config = dict(
    seed=2022,
    max_iters=total_step,
    eval_interval=10_000,
    log_interval=1_000,
    fp16=True,
    threshold=0.5
)

work_dir = f'./work_dir'

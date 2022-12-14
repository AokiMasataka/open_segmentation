model = dict(
    segmenter=dict(type='EncoderDecoder'),
    backbone=dict(
        type='ConvNeXt',
        depth=(3, 3, 27, 3),
        dims=(128, 256, 512, 1024),
        out_indices=(0, 1, 2, 3),
        drop_path_rate=0.3,
        init_config=dict(pretrained='https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_384.pth')
    ),
    decoder=dict(
        type='UnetHead',
        encoder_channels=(128, 256, 512, 1024),
        decoder_channels=(512, 256, 128, 32),
        num_blocks=4,
        center_block_config=dict(type='CenterBlock', in_dim=1024),
        layers=1,
        eps=1e-6,
    ),
    loss=[dict(type='CrossEntropyLoss', mode='multiclass', label_smooth=0.01, loss_weight=1.0)],
    init_config=None,
    test_config=dict(mode='whole'),
    norm_config=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    num_classes=10
)


train_pipeline = [
    dict(type='LoadImageFromFile', to_float=True, color_type='anydepth', max_value='max', force_3chan=False),
    dict(type='LoadAnnotations'),
    dict(type='Resize', size=256, keep_retio=True),
    dict(type='RandomFlipHorizontal', prob=0.5),
    dict(type='Padding', pad_value=0, label_pad_value=255),
]


test_pipeline = [
    dict(type='LoadImageFromFile', to_float=True, color_type='anydepth', max_value='max', force_3chan=False),
    dict(type='LoadAnnotations'),
    dict(type='SimpleInferencer', input_size=256, keep_retio=True)
]


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

total_step = 60_000
optimizer = dict(type='AdamW', base_lr=1e-4, head_lr=2e-4, betas=(0.9, 0.999), weight_decay=1e-6)
scheduler = dict(type='CosineAnnealingLR', T_max=total_step, eta_min=5e-7)

train_config = dict(
    seed=2022,
    max_iters=total_step,
    eval_interval=10_000,
    log_interval=1_000,
    save_checkpoint=True,
    fp16=True,
    threshold=0.5,
)

work_dir = f'./work_dir'

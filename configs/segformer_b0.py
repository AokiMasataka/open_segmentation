model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='MixVisionTransformer',
        embed_dims=(32, 64, 160, 256),
        num_heads=(1, 2, 5, 8),
        mlp_ratios=(4, 4, 4, 4),
        qkv_bias=True,
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        drop_rate=0.0,
        drop_path_rate=0.1,
        init_config=dict(pretrained='https://github.com/qubvel/segmentation_models.pytorch/releases/download/v0.0.2/mit_b0.pth')
    ),
    decoder=dict(
        type='SegFormerHead',
        num_classes=151,
        input_dims=(32, 64, 160, 256),
        embedding_dim=256,
        drop_prob=0.0
    ),
    loss=[dict(type='CrossEntropyLoss', mode='multiclass', label_smooth=0.0, loss_weight=1.0)],
    init_config=None,
    test_config=dict(mode='whole'),
    norm_config=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
)

train_pipeline = [
    dict(type='LoadImageFromFile', to_float=True, color_type='anydepth', force_3chan=False),
    dict(type='LoadAnnotations'),
    dict(type='Resize', size=256, keep_retio=True),
    dict(type='RandomFlipHorizontal', prob=0.5),
    dict(type='Padding', pad_value=0, label_pad_value=255),
]


test_pipeline = [
    dict(type='LoadImageFromFile', to_float=True, color_type='anydepth', force_3chan=False),
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
optimizer = dict(type='AdamW', lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-6)
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

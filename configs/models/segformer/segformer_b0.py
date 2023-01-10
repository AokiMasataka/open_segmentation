model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='MixVisionTransformer',
        embed_dims=(32, 64, 160, 256),
        num_heads=(1, 2, 5, 8),
        mlp_ratios=(4, 4, 4, 4),
        depths=(2, 2, 2, 2),
        sr_ratios=(8, 4, 2, 1),
        qkv_bias=True,
        drop_rate=0.0,
        drop_path_rate=0.1,
        init_config=dict(pretrained='https://github.com/qubvel/segmentation_models.pytorch/releases/download/v0.0.2/mit_b0.pth')
    ),
    decoder=dict(
        type='SegFormerHead',
        num_classes=151,
        encoder_dims=(32, 64, 160, 256),
        embedding_dim=256,
        drop_prob=0.0
    ),
    loss=[dict(type='DiceLoss', loss_weight=1.0)],
    init_config=None,
    test_config=dict(mode='whole'),
    norm_config=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], div=255.0),
)

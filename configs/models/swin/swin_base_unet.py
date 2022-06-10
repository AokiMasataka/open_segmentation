model = dict(
    segmenter=dict(type='EncoderDecoder'),
    backbone=dict(
        type='SwinTransformer',
        patch_size=4,
        window_size=12,
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        init_config=dict(
            pretrained='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth'
        )
    ),
    decoder=dict(
        type='UnetHead',
        num_classes=10,
        encoder_dims=(128, 256, 512, 1024),
        decoder_dims=(128, 256, 512, 1024),
        center_block_config=dict(type='CenterBlock', input_dim=1024),
        layers=1,
    ),
    loss=[dict(type='CrossEntropyLoss', mode='multiclass', label_smooth=0.01, loss_weight=1.0)],
    init_config=None,
    test_config=dict(mode='whole'),
    norm_config=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    num_classes=10
)
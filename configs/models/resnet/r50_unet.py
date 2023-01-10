model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='ResNet',
        block_type='Bottleneck',
        layers=(3, 4, 6, 3),
        in_channels=3,
        stem_width=64,
        channels=(256, 512, 1024, 2048),
        act_config=dict(type='ReLU', inplace=True),
        init_config=dict(pretrained='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnet50_ra_224-8efdb4bb.pth')
    ),
    decoder=dict(
        type='UnetHead',
        num_classes=10,
        encoder_dims=(64, 256, 512, 1024, 2048),
        decoder_dims=(32, 64, 256, 512, 1024),
        center_block_config=dict(type='CenterBlock', input_dim=2048),
        layers=1,
    ),
    loss=[dict(type='DiceLoss', loss_weight=1.0)],
    init_config=None,
    test_config=dict(mode='whole'),
    norm_config=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], div=255.0)
)
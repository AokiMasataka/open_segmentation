_base_ = ['./configs/models/swin/swin_base_unet.py', './configs/pipeline/basic_pipeline.py']

total_step = 40_000
optimizer = dict(type='AdamW', base_lr=1e-4, head_lr=2e-4, betas=(0.9, 0.999), weight_decay=1e-6)
scheduler = dict(type='CosineAnnealingLR', T_max=total_step, eta_min=5e-7)
metrics = ['mdice', 'miou']

train_config = dict(
    seed=2022,
    max_iters=total_step,
    eval_interval=5000,
    log_interval=500,
    save_checkpoint=True,
    fp16=True,
    threshold=0.5,
)

work_dir = f'./works/'

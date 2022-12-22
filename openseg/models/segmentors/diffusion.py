import torch
from torch import nn
from .base import SegmentorBase
from ..builder import SEGMENTER, BACKBONES, DECODERS, LOSSES
from openseg.utils.torch_utils import force_fp32


@SEGMENTER.register_module
class Diffusion(SegmentorBase):
    def __init__(
        self,
        backbone: dict,
        decoder: dict,
        loss: dict,
        test_config: dict = None,
        init_config: dict = None,
        norm_config: dict = None,
        timesteps: int = 200,
    ):
        super(Diffusion, self).__init__(
            num_classes=decoder['num_calsses'],
            test_config=test_config,
            init_config=init_config,
            norm_config=norm_config
        )
        self.maxtime = timesteps
        self.backbone = BACKBONES.build(config=backbone)
        self.decoder = DECODERS.build(config=decoder)
        self.losses = LOSSES.build(config=loss)

        self.noise_generator = NoiseGenerator(timesteps=timesteps)

        self.init()
    
    def forward(self, images, num_steps=10):
        noise = torch.randn_like(images)
        feats = self.backbone(images)
        for i in reversed(range(num_steps)):
            timestep = torch.tensor([i], dtype=torch.long, device=images.device)
            denoise = self.decoder(hidden_state=noise, context=feats[0], timestep=timestep)
            noise = noise - denoise
        return noise

    def forward_train(self, images, labels):
        timestep = torch.randint(low=0, high=self.maxtime, size=images.shape[0])
        noisy_label, noise = self.noise_generator(images=labels, timestep=timestep)

        feats = self.backbone(images)
        logits = self.decoder(hidden_state=noisy_label, context=feats[0], timestep=timestep)

        loss, losses = self._get_loss(logits=logits, labels=noise)
        return loss, losses

    def forward_test(self, images, labels, num_steps=10):
        logits = self(images, num_steps)
        loss, _ = self._get_loss(logits=logits, labels=labels)
        return loss
    
    @force_fp32
    def _get_loss(self, logits, labels):
        losses = {}
        for loss_name, loss_fn in zip(self.losses.keys(), self.losses.values()):
            losses[loss_name] = loss_fn(logits, labels)
        loss = sum(losses.values())
        return loss, losses


class NoiseGenerator:
    def __init__(self, timesteps):
        self.timesteps = timesteps
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    def __call__(self, labels, timestep):
        noise = torch.randn_like(labels)
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, timestep)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, timestep)
        return sqrt_alphas_cumprod_t * labels + sqrt_one_minus_alphas_cumprod_t * noise, noise

    def extract(self, a, timestep):
        out = torch.gather(input=a, dim=-1, index=timestep.cpu())
        return out.view(timestep.shape[0], 1, 1, 1).to(timestep.device)
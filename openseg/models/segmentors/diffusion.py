import torch
from torch import nn
from .base import SegmentorBase
from ..builder import SEGMENTER, BACKBONES, DECODERS, LOSSES


@SEGMENTER.register_module
class Diffusion(SegmentorBase):
    def __init__(
        self,
        backbone: dict,
        decoder: dict,
        losses: dict,
        joiner: dict = dict(type='last_layer', input_dim=None, output_dim=None),
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
        self.losses = LOSSES.build(config=losses)
        self.noise_generator = NoiseGenerator(timesteps=timesteps)

        self.init()
    
    def forward_feature(self, images):
        images = self.norm_fn(images=images)
        features = self.backbone(images)
        context = self.somthing(features)
        return context

    def forward_train(self, images, labels):
        context = self.forward_feature(images=images)

        timestep = torch.randint(low=0, high=self.maxtime, size=images.shape[0])
        noisy_labels, noise = self.noise_generator(images=labels, timestep=timestep)

        logits = self.decoder(hidden_state=noisy_labels, context=context, timestep=timestep)

        loss, losses = self._get_loss(logits=logits, labels=noise)
        return loss, losses

    def forward_test(self, images, labels, num_steps=10):
        logits = self(images, num_steps)
        loss, _ = self._get_loss(logits=logits, labels=labels)
        return loss


class NoiseGenerator(nn.Module):
    def __init__(self, timesteps):
        super(NoiseGenerator, self).__init__()
        self.timesteps = timesteps
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

    def __call__(self, labels, timestep):
        noise = torch.randn_like(labels)
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, timestep)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, timestep)
        return sqrt_alphas_cumprod_t * labels + sqrt_one_minus_alphas_cumprod_t * noise, noise

    def extract(self, a, timestep):
        out = torch.gather(input=a, dim=-1, index=timestep.cpu())
        return out.view(timestep.shape[0], 1, 1, 1).to(timestep.device)
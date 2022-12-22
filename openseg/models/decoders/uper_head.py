import torch
from torch import nn
from torch.nn import functional
from .base import DecoderBase
from ..builder import DECODERS


@DECODERS.register_module
class UperHead(DecoderBase):
    def __init__(
        self,
        num_classes: int,
        encoder_dims: tuple,
        output_dim: int,
        pool_scales: tuple = (1, 2, 3, 6),
        eps: float = 1e-6,
        fpn_bottleneck_type: str = 'conv'
    ):
        super(UperHead, self).__init__()
        assert fpn_bottleneck_type in ('conv', 'attention')
        self.encoder_dims = encoder_dims
        self.output_dim = output_dim
        self.psp_modules = PPM(pool_scales, encoder_dims[-1], output_dim, eps)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels=encoder_dims[-1] + len(pool_scales) * output_dim,
                out_channels=output_dim,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1
            ),
            nn.BatchNorm2d(num_features=output_dim, eps=eps),
            nn.ReLU(inplace=True)
        )

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in encoder_dims[:-1]:  # skip the top layer
            l_conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=output_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
                nn.BatchNorm2d(num_features=output_dim, eps=eps),
                nn.ReLU(inplace=True)
            )
            fpn_conv = nn.Sequential(
                nn.Conv2d(in_channels=output_dim, out_channels=output_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
                nn.BatchNorm2d(num_features=output_dim, eps=eps),
                nn.ReLU(inplace=True)
            )
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
        
        bottleneck_channels = len(self.encoder_dims) * output_dim

        if fpn_bottleneck_type == 'conv':
            self.fpn_bottleneck = nn.Sequential(
                nn.Conv2d(in_channels=bottleneck_channels, out_channels=output_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
                nn.BatchNorm2d(num_features=output_dim, eps=eps),
                nn.ReLU(inplace=True)
            )
        elif fpn_bottleneck_type == 'attention':
            # TODO self.fpn_bottleneck to Window-Attention
            pass
        
        self.head = nn.Sequential(
            nn.Conv2d(output_dim, output_dim * 4, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.GELU(),
            nn.Dropout2d(p=0.1),
            nn.BatchNorm2d(num_features=output_dim * 4, eps=eps),
            nn.Conv2d(output_dim * 4, num_classes, kernel_size=(1, 1), stride=(1, 1), padding=0),
        )
    
    def psp_forward(self, features):
        """Forward function of PSP module."""
        x = features[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        return self.bottleneck(psp_outs)
    
    def forward(self, features):
        laterals = [lateral_conv(features[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        laterals.append(self.psp_forward(features))

        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + functional.interpolate(
                laterals[i], size=prev_shape, mode='bilinear', align_corners=self.align_corners
            )
        
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels - 1)]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = functional.interpolate(
                fpn_outs[i], size=fpn_outs[0].shape[2:], mode='bilinear', align_corners=self.align_corners
            )
        fpn_outs = torch.cat(fpn_outs, dim=1)
        return self.head(self.fpn_bottleneck(fpn_outs))


class PPM(nn.ModuleList):
    def __init__(self, pool_scales, in_dims, channels, eps):
        super(PPM, self).__init__()
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    nn.Conv2d(in_dims, channels, kernel_size=(1, 1), stride=(1, 1), bias=False),
                    nn.BatchNorm2d(num_features=channels, eps=eps),
                    nn.ReLU(inplace=True)
                )
            )

    def forward(self, x):
        ppm_outs = list()
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = functional.interpolate(ppm_out, size=x.size()[2:], mode='bilinear', align_corners=False)
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs
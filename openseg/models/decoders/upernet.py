import torch
from torch import nn
from torch.nn import functional
from .base import DecoderBase
from ..builder import DECODERS


@DECODERS.register_module
class Upernet(DecoderBase):
    def __init__(self, encoder_channels, channels, pool_scales=(1, 2, 3, 6), align_corners=False, fpn_bottleneck_type='conv'):
        super(Upernet, self).__init__()
        assert fpn_bottleneck_type in ('conv', 'attention')
        self._encoder_channels = encoder_channels
        self.align_corners = align_corners
        self.channels = channels
        self.psp_modules = PPM(
            pool_scales, encoder_channels[-1], channels, align_corners=self.align_corners
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels=encoder_channels[-1] + len(pool_scales) * channels,
                out_channels=channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1
            ),
            nn.BatchNorm2d(num_features=channels),
            nn.ReLU(inplace=True)
        )

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in encoder_channels[:-1]:  # skip the top layer
            l_conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
                nn.BatchNorm2d(num_features=channels),
                nn.ReLU(inplace=True)
            )
            fpn_conv = nn.Sequential(
                nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
                nn.BatchNorm2d(num_features=channels),
                nn.ReLU(inplace=True)
            )
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
        
        bottleneck_channels = len(self.encoder_channels) * channels

        if fpn_bottleneck_type == 'conv':
            self.fpn_bottleneck = nn.Sequential(
                nn.Conv2d(in_channels=bottleneck_channels, out_channels=channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
                nn.BatchNorm2d(num_features=channels),
                nn.ReLU(inplace=True)
            )
        elif fpn_bottleneck_type == 'attention':
            # TODO self.fpn_bottleneck to Window-Attention
            pass
    
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
        return self.fpn_bottleneck(fpn_outs)
    
    def decoder_out_dim(self):
        return self.channels


class PPM(nn.ModuleList):
    def __init__(self, pool_scales, in_channels, channels, align_corners):
        super(PPM, self).__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    nn.Conv2d(in_channels=self.in_channels, out_channels=self.channels, kernel_size=(1, 1), stride=(1, 1), bias=False),
                    nn.BatchNorm2d(num_features=self.channels),
                    nn.ReLU(inplace=True)
                )
            )

    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = functional.interpolate(
                ppm_out, size=x.size()[2:], mode='bilinear', align_corners=self.align_corners
            )
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs
import torch
from torch import nn
from torch.nn import functional


class DecoderBase(nn.Module):
    def __init__(self):
        super(DecoderBase, self).__init__()
        pass

    def decoder_out_dim(self):
        return self._decoder_channels[-1]


class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=(1, 1)),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=(1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class Attention(nn.Module):
    def __init__(self, name, **params):
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)
        elif name == "scse":
            self.attention = SCSEModule(**params)
        else:
            raise ValueError(f'Attention {name} is not implemented')

    def forward(self, x):
        return self.attention(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, scale_factor=2, mode='nearest', attention_type=None):
        super(DecoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.attention1 = Attention(attention_type, in_channels=in_ch)
        self.attention2 = Attention(attention_type, in_channels=out_ch)

        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x, skip=None):
        x = functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        if skip is not None:
            x = torch.cat(tensors=(x, skip), dim=1)
            x = self.attention1(x)
        x = self.block(x)
        return self.attention2(x)


class ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, groups=1):
        super(ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(
            in_channels=inplanes,
            out_channels=planes,
            kernel_size=(kernel_size, kernel_size),
            stride=(1, 1),
            padding=padding,
            dilation=dilation,
            bias=False,
            groups=groups
        )
        self.bn = nn.BatchNorm2d(planes)
        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        return functional.relu(x, inplace=True)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, in_channels=512, out_channels=512, dilations=(1, 2, 4, 8)):
        super(ASPP, self).__init__()
        mid_channels = out_channels // 4
        self.aspps = [ASPPModule(in_channels, mid_channels, 1, padding=0, dilation=1)]
        self.aspps += [ASPPModule(in_channels, mid_channels, 3, padding=d, dilation=d, groups=4) for d in dilations]
        self.aspps = nn.ModuleList(self.aspps)
        self.global_pool = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        in_channels = mid_channels * (2 + len(dilations))
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self._init_weight()

    def forward(self, x):
        x0 = self.global_pool(x)
        xs = [aspp(x) for aspp in self.aspps]
        x0 = functional.interpolate(x0, size=xs[0].size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x0] + xs, dim=1)
        return self.out_conv(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1,
            bias=False
        )
        bn1 = nn.BatchNorm2d(out_channels)
        activate1 = nn.ReLU()

        conv2 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1,
            bias=False
        )
        bn2 = nn.BatchNorm2d(out_channels)
        activate2 = nn.ReLU()

        super().__init__(conv1, bn1, activate1, conv2, bn2, activate2)

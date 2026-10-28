"""SpatialUNet encoder adapted to RT-DETR's S2-S5 feature interface.

Retains the supplied encoder (Conv/GN/GELU, widths 16..256), adds a
stride-32 stage and omits the reconstruction decoder and pooled latent head.
This is an encoder-only adaptation, not a full U-Net detection neck.
"""
import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.layers(x)


class SpatialUNetBackbone(nn.Module):
    def __init__(self, input_channels=3, base_channels=16, use_p2=False,
                 use_spd=False, use_s2_fusion=False, spd_factory=None):
        super().__init__()
        if base_channels <= 0 or base_channels % 8:
            raise ValueError('base_channels must be a positive multiple of 8')
        if use_s2_fusion and (use_p2 or use_spd):
            raise ValueError('S2 fusion cannot be combined with P2 or global SPD')
        if use_spd and spd_factory is None:
            raise ValueError('SPD mode requires an SPDConv factory')
        self.input_channels = input_channels
        b = base_channels
        self.encoder1 = ConvBlock(input_channels, b)
        self.encoder2 = ConvBlock(b, 2*b)
        self.encoder3 = ConvBlock(2*b, 4*b)
        self.encoder4 = ConvBlock(4*b, 8*b)
        self.bottleneck = ConvBlock(8*b, 16*b)
        self.encoder5 = ConvBlock(16*b, 32*b)
        # Inject the existing RT-DETR SPDConv to avoid duplicate implementations
        # and a circular import. Each transition has independent parameters.
        self.downsamples = nn.ModuleList([
            spd_factory(c, c, 3, activation='gelu') if use_spd
            else nn.MaxPool2d(2, 2)
            for c in (b, 2*b, 4*b, 8*b, 16*b)
        ])
        self.return_s2 = use_p2 or use_s2_fusion
        self.out_channels = (4*b, 8*b, 16*b, 32*b) if self.return_s2 else (8*b, 16*b, 32*b)
        self.out_strides = (4, 8, 16, 32) if self.return_s2 else (8, 16, 32)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.input_channels:
            raise ValueError(f'Expected [B,{self.input_channels},H,W], got {tuple(x.shape)}')
        if x.shape[-2] % 32 or x.shape[-1] % 32 or min(x.shape[-2:]) < 32:
            raise ValueError('RT-DETR SpatialUNet input dimensions must be positive multiples of 32')
        x = self.encoder1(x)
        x = self.encoder2(self.downsamples[0](x))
        s2 = self.encoder3(self.downsamples[1](x))
        s3 = self.encoder4(self.downsamples[2](s2))
        s4 = self.bottleneck(self.downsamples[3](s3))
        s5 = self.encoder5(self.downsamples[4](s4))
        return (s2, s3, s4, s5) if self.return_s2 else (s3, s4, s5)

"""3D UNet Generator for pix2pix-style CT-to-PET synthesis."""

import torch
import torch.nn as nn


class DownBlock3D(nn.Module):
    """Conv3d(k=4, s=2, p=1) → InstanceNorm → LeakyReLU(0.2)."""

    def __init__(self, in_ch, out_ch, norm=True):
        super().__init__()
        layers = [nn.Conv3d(in_ch, out_ch, 4, 2, 1, bias=not norm)]
        if norm:
            layers.append(nn.InstanceNorm3d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UpBlock3D(nn.Module):
    """Upsample(trilinear, ×2) → Conv3d(k=3) → InstanceNorm → ReLU.

    Uses trilinear upsample + conv instead of ConvTranspose3d to avoid
    checkerboard artifacts (Odena et al. 2016).
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet3DGenerator(nn.Module):
    """4-level 3D UNet with skip connections.

    Input:  (B, 1, Z, H, W)  — normalized CT
    Output: (B, 1, Z, H, W)  — synthetic PET in [-1, 1] (tanh)
    """

    def __init__(self, in_ch=1, out_ch=1, base=32):
        super().__init__()
        b = base

        self.down1 = DownBlock3D(in_ch, b,      norm=False)
        self.down2 = DownBlock3D(b,     b * 2)
        self.down3 = DownBlock3D(b * 2, b * 4)
        self.down4 = DownBlock3D(b * 4, b * 8)

        self.bottleneck = nn.Sequential(
            nn.Conv3d(b * 8, b * 8, 3, 1, 1),
            nn.InstanceNorm3d(b * 8),
            nn.ReLU(inplace=True),
        )

        self.up4 = UpBlock3D(b * 8,           b * 8)
        self.up3 = UpBlock3D(b * 8 + b * 4,   b * 4)
        self.up2 = UpBlock3D(b * 4 + b * 2,   b * 2)
        self.up1 = UpBlock3D(b * 2 + b,        b)

        self.final = nn.Sequential(
            nn.Conv3d(b + in_ch, out_ch, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        b = self.bottleneck(d4)

        u4 = torch.cat([self.up4(b),  d3], 1)
        u3 = torch.cat([self.up3(u4), d2], 1)
        u2 = torch.cat([self.up2(u3), d1], 1)
        u1 = torch.cat([self.up1(u2), x],  1)

        return self.final(u1)

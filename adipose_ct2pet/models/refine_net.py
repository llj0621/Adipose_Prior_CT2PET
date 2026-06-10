"""Lightweight 3-level UNet for residual refinement (Stage 2).

Supports four conditioning modes:
    M1  : in_ch=2  (fake_PET, CT)            — no adipose prior
    M2L : in_ch=2  (fake_PET, CT)            — adipose prior in loss only
    M2  : in_ch=4  (fake_PET, CT, VAT, SAT)  — adipose prior as input
    M3  : in_ch=4  (fake_PET, CT, VAT, SAT)  — adipose prior as input + in loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, 1, 1),
            nn.InstanceNorm3d(out_ch), nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, 1, 1),
            nn.InstanceNorm3d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock3D(nn.Module):
    """Trilinear upsample + skip-cat + ConvBlock. Handles odd-sized feature maps."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(in_ch, out_ch, 3, 1, 1),
        )
        self.conv = ConvBlock3D(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # pad to match skip when spatial dims differ by 1 (rounding from stride-2 down)
        dZ = skip.size(2) - x.size(2)
        dY = skip.size(3) - x.size(3)
        dX = skip.size(4) - x.size(4)
        x = F.pad(x, [dX // 2, dX - dX // 2,
                       dY // 2, dY - dY // 2,
                       dZ // 2, dZ - dZ // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class RefinementUNet3D(nn.Module):
    """Residual refinement UNet.  Output = Stage-1 fake + learned residual.

    Zero-initialised output conv so the network starts as an identity map.
    """

    def __init__(self, in_ch=2, base_ch=32):
        super().__init__()
        b = base_ch
        self.enc1  = ConvBlock3D(in_ch, b)
        self.down1 = nn.Conv3d(b, b * 2, 2, 2)
        self.enc2  = ConvBlock3D(b * 2, b * 2)
        self.down2 = nn.Conv3d(b * 2, b * 4, 2, 2)
        self.enc3     = ConvBlock3D(b * 4, b * 4)
        self.up2      = UpBlock3D(b * 4, b * 2)
        self.up1      = UpBlock3D(b * 2, b)
        self.out_conv = nn.Conv3d(b, 1, 1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, *inputs):
        """inputs: (fake_PET, CT [, torso_mask, subcu_mask])."""
        fake = inputs[0]
        x    = torch.cat(inputs, dim=1)
        e1   = self.enc1(x)
        e2   = self.enc2(self.down1(e1))
        b    = self.enc3(self.down2(e2))
        d1   = self.up1(self.up2(b, e2), e1)
        return fake + self.out_conv(d1)

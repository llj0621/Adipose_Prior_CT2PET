"""3D PatchGAN discriminator with Spectral Norm and multi-scale wrapper."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


def _sn_conv(in_ch, out_ch, k, s, p, bias=False):
    return spectral_norm(nn.Conv3d(in_ch, out_ch, k, s, p, bias=bias))


class PatchGAN3D(nn.Module):
    """Single-scale 3D PatchGAN with Spectral Normalization (Miyato et al. 2018).

    Input:  concatenated (CT, PET) — 2 channels.
    Output: logit map [B, 1, D', H', W'].
    """

    def __init__(self, in_ch=1, out_ch=1, base_ch=64, final_kernel=4):
        super().__init__()
        c = base_ch
        # Named attributes match original state dict keys (blk0..blk3, final_conv)
        self.blk0 = nn.Sequential(
            _sn_conv(in_ch + out_ch, c, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.blk1 = nn.Sequential(
            _sn_conv(c, c * 2, 4, 2, 1),
            nn.InstanceNorm3d(c * 2), nn.LeakyReLU(0.2, inplace=True),
        )
        self.blk2 = nn.Sequential(
            _sn_conv(c * 2, c * 4, 4, 2, 1),
            nn.InstanceNorm3d(c * 4), nn.LeakyReLU(0.2, inplace=True),
        )
        self.blk3 = nn.Sequential(
            _sn_conv(c * 4, c * 8, 4, 1, 1),   # stride=1
            nn.InstanceNorm3d(c * 8), nn.LeakyReLU(0.2, inplace=True),
        )
        self.final_conv = _sn_conv(c * 8, 1, final_kernel, 1, final_kernel // 2, bias=True)
        # Register as a list for iteration in forward(); named attrs handle state dict
        self._blocks = [self.blk0, self.blk1, self.blk2, self.blk3]

    def forward(self, x, y, return_feats=False):
        h = torch.cat([x, y], dim=1)
        feats = []
        for blk in self._blocks:
            h = blk(h)
            feats.append(h)
        out = self.final_conv(h)
        return (out, feats) if return_feats else out


class MultiScalePatchGAN3D(nn.Module):
    """Two-scale PatchGAN: full-res + 2× downsampled inputs."""

    def __init__(self, in_ch=1, out_ch=1, base_ch=64,
                 num_scales=2, min_spatial=16):
        super().__init__()
        self.num_scales  = num_scales
        self.min_spatial = min_spatial
        self.discriminators = nn.ModuleList(
            [PatchGAN3D(in_ch, out_ch, base_ch) for _ in range(num_scales)]
        )

    def forward(self, x, y, return_feats=False):
        outputs, feats_list = [], []
        xs, ys = x, y
        for i, D in enumerate(self.discriminators):
            if min(xs.shape[2:]) < self.min_spatial:
                break
            if return_feats:
                o, f = D(xs, ys, return_feats=True)
                outputs.append(o)
                feats_list.append(f)
            else:
                outputs.append(D(xs, ys))
            if i != self.num_scales - 1:
                xs = F.avg_pool3d(xs, 2, 2)
                ys = F.avg_pool3d(ys, 2, 2)

        if not outputs:
            if return_feats:
                o, f = self.discriminators[0](x, y, return_feats=True)
                outputs.append(o); feats_list.append(f)
            else:
                outputs.append(self.discriminators[0](x, y))

        return (outputs, feats_list) if return_feats else outputs

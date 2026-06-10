"""Stable 3D Multi-Scale SSIM loss (Wang et al. 2003)."""

import torch
import torch.nn.functional as F


def _gaussian_1d(size=11, sigma=1.5, device="cpu"):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _gaussian_3d(size=11, sigma=1.5, device="cpu"):
    g = _gaussian_1d(size, sigma, device)
    k = g[:, None, None] * g[None, :, None] * g[None, None, :]
    return k.unsqueeze(0).unsqueeze(0)   # (1, 1, D, H, W)


def _ssim3d(x, y, kernel, C1=1e-4, C2=9e-4):
    p = kernel.shape[-1] // 2
    ux  = F.conv3d(x,     kernel, padding=p)
    uy  = F.conv3d(y,     kernel, padding=p)
    uxx = F.conv3d(x * x, kernel, padding=p)
    uyy = F.conv3d(y * y, kernel, padding=p)
    uxy = F.conv3d(x * y, kernel, padding=p)
    vx  = (uxx - ux * ux).clamp_min(0)
    vy  = (uyy - uy * uy).clamp_min(0)
    vxy = uxy - ux * uy
    num = (2 * ux * uy + C1) * (2 * vxy + C2)
    den = (ux * ux + uy * uy + C1) * (vx + vy + C2)
    return (num / (den + 1e-8)).mean()


def ms_ssim_loss(x, y, levels=3, kernel_size=11, sigma=1.5):
    """1 − MS-SSIM for (B, 1, D, H, W) volumes. Lower is better."""
    k = _gaussian_3d(kernel_size, sigma, device=x.device)
    score, w = 1.0, 1.0 / levels
    for _ in range(levels):
        score *= _ssim3d(x, y, k).clamp(1e-6, 1.0) ** w
        x = F.avg_pool3d(x, 2, 2)
        y = F.avg_pool3d(y, 2, 2)
    return 1.0 - score

"""Shared utility functions."""

import os
import numpy as np
import torch


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def next_exp_dir(base_dir):
    """Auto-increment experiment directory: base_dir/1, /2, /3, ..."""
    ensure_dir(base_dir)
    existing = [int(n) for n in os.listdir(base_dir) if n.isdigit()]
    idx = max(existing) + 1 if existing else 1
    path = os.path.join(base_dir, str(idx))
    ensure_dir(path)
    return path


def amp_ok(device, flag):
    """AMP is only valid on CUDA."""
    dev = device.type if isinstance(device, torch.device) else str(device)
    return bool(flag) and dev == "cuda"


# ── PET normalisation ─────────────────────────────────────────

def denorm_pet(x, pet_clip_high=5.0):
    """[-1, 1] → [0, pet_clip_high] SUV (numpy or torch)."""
    return (x + 1.0) * (pet_clip_high / 2.0)


def norm_pet(x, pet_clip_high=5.0):
    """[0, pet_clip_high] SUV → [-1, 1]."""
    return x / (pet_clip_high / 2.0) - 1.0


# ── Overlap-add blending window ───────────────────────────────

def gaussian_weights(length, sigma=8):
    """1-D Gaussian weight tensor for patch overlap blending (peak = 1)."""
    coords = torch.arange(length, dtype=torch.float32)
    w = torch.exp(-((coords - (length - 1) / 2.0) ** 2) / (2 * sigma ** 2))
    return w / w.max()

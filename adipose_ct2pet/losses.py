"""Loss functions: adversarial, L1, feature matching, fat-stratified."""

import torch
import torch.nn as nn


# ── Adversarial ────────────────────────────────────────────────

def d_loss(d_real, d_fake):
    """Hinge-free BCE discriminator loss, averaged over scales."""
    bce = nn.BCEWithLogitsLoss()

    def _one(dr, df):
        return 0.5 * (bce(dr, torch.ones_like(dr)) +
                      bce(df, torch.zeros_like(df)))

    if isinstance(d_real, (list, tuple)):
        return sum(_one(r, f) for r, f in zip(d_real, d_fake)) / len(d_real)
    return _one(d_real, d_fake)


def g_adv_loss(d_fake):
    bce = nn.BCEWithLogitsLoss()
    if isinstance(d_fake, (list, tuple)):
        return sum(bce(df, torch.ones_like(df)) for df in d_fake) / len(d_fake)
    return bce(d_fake, torch.ones_like(d_fake))


# ── Reconstruction ─────────────────────────────────────────────

def l1(a, b):
    return torch.mean(torch.abs(a - b))


def masked_l1(a, b, mask, eps=1e-8):
    """L1 restricted to voxels where mask == 1."""
    return (torch.abs(a - b) * mask).sum() / (mask.sum() + eps)


def feature_matching_loss(real_feats, fake_feats):
    """L1 distance between intermediate discriminator feature maps."""
    total = sum(torch.mean(torch.abs(r - f))
                for r, f in zip(real_feats, fake_feats))
    return total / max(1, len(real_feats))


# ── Adipose-stratified losses (M2L / M3) ──────────────────────

def fat_stratified_losses(fake, real, torso_mask, subcu_mask):
    """Return VAT and SAT masked L1 as a dict."""
    return {
        "torso": masked_l1(fake, real, torso_mask),
        "subcu": masked_l1(fake, real, subcu_mask),
    }


def get_fat_weights(epoch, num_epochs, w_torso, w_subcu, max_scale=1.0):
    """Linearly ramp base weights from 1× → max_scale× over training.

    max_scale=1.0 → static weights (no ramp).
    """
    t = (epoch - 1) / max(1, num_epochs - 1)
    scale = 1.0 + t * (max_scale - 1.0)
    return w_torso * scale, w_subcu * scale

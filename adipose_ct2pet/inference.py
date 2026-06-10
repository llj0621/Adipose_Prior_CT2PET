"""Sliding-window inference with Gaussian overlap fusion."""

import os
import numpy as np
import nibabel as nib
import torch
from scipy.ndimage import zoom

from adipose_ct2pet.dataset import PatchDataset, cached_patches
from adipose_ct2pet.utils   import gaussian_weights, denorm_pet, ensure_dir, amp_ok
import adipose_ct2pet.config as cfg


def infer(G, R, mode, case_id, device,
          output_dir=None, save_nifti=True, cache=None):
    """Reconstruct full PET volume for one case via sliding-window fusion.

    Args:
        G:          UNet3DGenerator (Stage 1).
        R:          RefinementUNet3D or None (M0 skips Stage 2).
        mode:       "M0" | "M1" | "M2" | "M2L" | "M3"
        case_id:    subject identifier string.
        device:     torch.device.
        output_dir: if set and save_nifti=True, writes {case_id}_PET_{mode}.nii.gz.
        cache:      optional volume cache dict from build_volume_cache().

    Returns:
        (pet_suv: np.ndarray [Z, H, W],  affine: np.ndarray [4, 4])
    """
    if cache and case_id in cache:
        patches = cached_patches(cache[case_id], cfg.PATCH_Z, cfg.PATCH_STRIDE)
    else:
        ds = PatchDataset(
            cfg.DATA_DIR, [case_id],
            target_xy=cfg.TARGET_XY, patch_z=cfg.PATCH_Z,
            patch_stride=cfg.PATCH_STRIDE,
            load_masks=(mode in ("M2", "M3")),
        )
        patches = ds.all_patches(case_id)

    G.eval()
    if R is not None:
        R.eval()

    ref_nii = nib.load(os.path.join(cfg.DATA_DIR, case_id, "CT.nii.gz"))
    z_all   = ref_nii.shape[2]
    orig_xy = ref_nii.shape[0]

    vol_sum = np.zeros((z_all, cfg.TARGET_XY, cfg.TARGET_XY), dtype=np.float32)
    wgt_sum = np.zeros_like(vol_sum)
    weights = gaussian_weights(cfg.PATCH_Z).numpy()
    use_amp = amp_ok(device, cfg.USE_AMP)

    for z0, batch in patches:
        ct = batch["ct"].unsqueeze(0).to(device)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=use_amp):
                fake = G(ct).float()
                if R is not None:
                    if mode in ("M2", "M3"):
                        fake = R(fake, ct,
                                 batch["torso_mask"].unsqueeze(0).to(device),
                                 batch["subcu_mask"].unsqueeze(0).to(device))
                    else:
                        fake = R(fake, ct)

        pred = fake[0, 0].cpu().numpy()
        vol_sum[z0:z0 + cfg.PATCH_Z] += pred * weights[:, None, None]
        wgt_sum[z0:z0 + cfg.PATCH_Z] += weights[:, None, None]

    fused   = vol_sum / np.maximum(wgt_sum, 1e-8)
    pet_suv = denorm_pet(fused, cfg.PET_CLIP[1])

    if cfg.TARGET_XY != orig_xy:
        scale   = orig_xy / cfg.TARGET_XY
        pet_suv = zoom(pet_suv, (1.0, scale, scale), order=1)

    if save_nifti and output_dir:
        ensure_dir(output_dir)
        out_data = np.transpose(pet_suv, (2, 1, 0)).astype(np.float32)  # → (X,Y,Z)
        nib.save(
            nib.Nifti1Image(out_data, ref_nii.affine),
            os.path.join(output_dir, f"{case_id}_PET_{mode}.nii.gz"),
        )

    return pet_suv, ref_nii.affine

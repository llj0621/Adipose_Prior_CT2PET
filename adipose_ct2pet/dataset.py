"""Dataset and volume-cache utilities for 3D CT/PET patch sampling."""

import math
import os
import numpy as np
import nibabel as nib
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset


def _num_patches(z_len, patch_z, stride):
    """Number of sliding-window patches with right-aligned tail coverage."""
    if z_len < patch_z:
        return 0
    if z_len == patch_z:
        return 1
    return math.ceil((z_len - patch_z) / stride) + 1


class PatchDataset(Dataset):
    """Z-axis sliding-window dataset for 3D volumes.

    Each __getitem__ returns one randomly-sampled patch along Z (training).
    Use all_patches(case_id) for deterministic inference / validation.

    Expected directory layout::

        DATA_DIR/{case_id}/CT.nii.gz
        DATA_DIR/{case_id}/PET.nii.gz
        DATA_DIR/{case_id}/torso_fat.nii.gz
        DATA_DIR/{case_id}/subcutaneous_fat.nii.gz
    """

    def __init__(self, data_dir, case_ids,
                 target_xy=256, patch_z=32, patch_stride=16,
                 ct_clip=(-1000, 1000), pet_clip=(0, 5),
                 load_masks=True):
        self.data_dir    = data_dir
        self.target_xy   = int(target_xy)
        self.patch_z     = int(patch_z)
        self.stride      = int(patch_stride)
        self.ct_clip     = ct_clip
        self.pet_clip    = pet_clip
        self.load_masks  = load_masks

        # Filter to cases with valid files and sufficient Z depth
        self.case_ids = []
        for cid in sorted(case_ids):
            ct_path  = os.path.join(data_dir, cid, "CT.nii.gz")
            pet_path = os.path.join(data_dir, cid, "PET.nii.gz")
            if not (os.path.exists(ct_path) and os.path.exists(pet_path)):
                continue
            z = nib.load(ct_path).shape[2]
            if _num_patches(z, patch_z, patch_stride) > 0:
                self.case_ids.append(cid)

    def __len__(self):
        return len(self.case_ids)

    # ── I/O helpers ───────────────────────────────────────────

    def _load_nii(self, path):
        data = nib.load(path).get_fdata().astype(np.float32)
        if data.ndim == 4:
            data = data[..., 0]
        return np.transpose(data, (2, 1, 0))  # (X, Y, Z) → (Z, Y, X)

    def _resize_xy(self, vol, target, order=1):
        if vol.shape[1] == target and vol.shape[2] == target:
            return vol
        f = (1.0, target / vol.shape[1], target / vol.shape[2])
        return zoom(vol, f, order=order)

    # ── Case loading ──────────────────────────────────────────

    def _load_case(self, case_id):
        case_dir = os.path.join(self.data_dir, case_id)
        ct  = self._load_nii(os.path.join(case_dir, "CT.nii.gz"))
        pet = self._load_nii(os.path.join(case_dir, "PET.nii.gz"))

        # Normalise: CT → [0, 1],  PET → [-1, 1]
        ct  = np.clip(ct,  *self.ct_clip)  / float(self.ct_clip[1])
        pet = np.clip(pet, *self.pet_clip) / (self.pet_clip[1] / 2.0) - 1.0

        ct  = self._resize_xy(ct,  self.target_xy, order=1)
        pet = self._resize_xy(pet, self.target_xy, order=1)

        masks = None
        if self.load_masks:
            masks = []
            for fname in ("torso_fat.nii.gz", "subcutaneous_fat.nii.gz"):
                mp = os.path.join(case_dir, fname)
                if os.path.exists(mp):
                    m = self._resize_xy(self._load_nii(mp), self.target_xy, order=0)
                    masks.append((m > 0).astype(np.float32))
                else:
                    masks.append(np.zeros_like(pet, dtype=np.float32))

        return ct.astype(np.float32), pet.astype(np.float32), masks

    # ── Patch extraction ──────────────────────────────────────

    def _make_patch(self, ct, pet, masks, z_start):
        z_end = z_start + self.patch_z
        out = {
            "ct":  torch.from_numpy(ct[z_start:z_end].copy()).unsqueeze(0),
            "pet": torch.from_numpy(pet[z_start:z_end].copy()).unsqueeze(0),
        }
        if masks is not None:
            out["torso_mask"] = torch.from_numpy(masks[0][z_start:z_end].copy()).unsqueeze(0)
            out["subcu_mask"] = torch.from_numpy(masks[1][z_start:z_end].copy()).unsqueeze(0)
        return out

    def z_size(self, case_id):
        """Return Z depth of a case (reads header only, no full load)."""
        path = os.path.join(self.data_dir, case_id, "CT.nii.gz")
        return nib.load(path).shape[2]

    def __getitem__(self, idx):
        ct, pet, masks = self._load_case(self.case_ids[idx])
        z_max   = ct.shape[0] - self.patch_z
        z_start = np.random.randint(0, max(1, z_max + 1))
        return self._make_patch(ct, pet, masks, z_start)

    def all_patches(self, case_id):
        """Yield (z_start, patch_dict) for every patch (validation / inference)."""
        ct, pet, masks = self._load_case(case_id)
        z_all = ct.shape[0]
        num   = _num_patches(z_all, self.patch_z, self.stride)
        for i in range(num):
            z_start = i * self.stride
            z_end   = z_start + self.patch_z
            if z_end > z_all:            # right-align last patch
                z_start = z_all - self.patch_z
            yield z_start, self._make_patch(ct, pet, masks, z_start)


# ── Volume cache (avoid repeated disk I/O during val / test) ─

def build_volume_cache(dataset, verbose=True):
    """Pre-load all cases as fp16/uint8 arrays."""
    ids = dataset.case_ids
    if verbose:
        from tqdm import tqdm
        ids = tqdm(ids, desc="Building cache", leave=False)
    cache = {}
    for cid in ids:
        ct, pet, masks = dataset._load_case(cid)
        cache[cid] = {
            "ct":         ct.astype("float16"),
            "pet":        pet.astype("float16"),
            "torso_mask": masks[0].astype("uint8") if masks else None,
            "subcu_mask": masks[1].astype("uint8") if masks else None,
        }
    return cache


def cached_patches(entry, patch_z, stride):
    """Yield (z_start, patch_dict) from an in-memory cache entry."""
    ct, pet = entry["ct"], entry["pet"]
    mt, ms  = entry.get("torso_mask"), entry.get("subcu_mask")
    z_all   = ct.shape[0]
    num     = _num_patches(z_all, patch_z, stride)
    for i in range(num):
        z_start = i * stride
        z_end   = z_start + patch_z
        if z_end > z_all:
            z_start = z_all - patch_z
        s = slice(z_start, z_start + patch_z)
        out = {
            "ct":  torch.from_numpy(ct[s].astype("float32")).unsqueeze(0),
            "pet": torch.from_numpy(pet[s].astype("float32")).unsqueeze(0),
        }
        if mt is not None:
            out["torso_mask"] = torch.from_numpy(mt[s].astype("float32")).unsqueeze(0)
            out["subcu_mask"] = torch.from_numpy(ms[s].astype("float32")).unsqueeze(0)
        yield z_start, out

"""Test-set evaluation: whole-body + VAT/SAT stratified metrics."""

import os
import csv
import numpy as np
import nibabel as nib
import torch
from scipy.ndimage import zoom
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr

from adipose_ct2pet.inference import infer
from adipose_ct2pet.losses    import masked_l1
from adipose_ct2pet.ms_ssim   import ms_ssim_loss
import adipose_ct2pet.config as cfg


def _load_gt(case_id):
    """Load ground-truth PET (SUV) and fat masks aligned to PET space."""
    def _to_zyx(nii):
        data = nii.get_fdata().astype(np.float32)
        if data.ndim == 4:
            data = data[..., 0]
        return np.transpose(data, (2, 1, 0))

    pet = _to_zyx(nib.load(os.path.join(cfg.DATA_DIR, case_id, "PET.nii.gz")))
    masks = {}
    for name, fname in [("torso", "torso_fat.nii.gz"),
                         ("subcu", "subcutaneous_fat.nii.gz")]:
        mp = os.path.join(cfg.DATA_DIR, case_id, fname)
        if os.path.exists(mp):
            m = _to_zyx(nib.load(mp))
            if m.shape != pet.shape:
                f = tuple(t / s for t, s in zip(pet.shape, m.shape))
                m = zoom(m, f, order=0)
            masks[name] = (m > 0).astype(np.float32)
        else:
            masks[name] = np.zeros_like(pet, dtype=np.float32)
    return pet, masks


def evaluate(G, R, mode, test_ids, device, output_dir, cache=None):
    """Run evaluation for one mode.

    Returns:
        summary (dict), per_case_rows (list of dict)
    """
    os.makedirs(output_dir, exist_ok=True)
    rows = []

    for cid in tqdm(test_ids, desc=f"Eval {mode}"):
        pred, _ = infer(G, R, mode, cid, device,
                        output_dir=output_dir, save_nifti=True, cache=cache)

        gt, masks = _load_gt(cid)
        gt   = np.clip(gt, *cfg.PET_CLIP)

        if pred.shape != gt.shape:
            factors = tuple(t / s for t, s in zip(gt.shape, pred.shape))
            pred    = zoom(pred, factors, order=1)

        whole_l1   = float(np.mean(np.abs(pred - gt)))
        whole_psnr = float(psnr(gt, pred, data_range=cfg.PET_CLIP[1]))

        def _ssim(dev):
            half = cfg.PET_CLIP[1] / 2.0
            p = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).float().to(dev)
            r = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0).float().to(dev)
            with torch.no_grad():
                v = 1.0 - ms_ssim_loss(p / half - 1.0, r / half - 1.0).item()
            return v, p, r

        try:
            whole_ssim, p_t, r_t = _ssim(device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            whole_ssim, p_t, r_t = _ssim(torch.device("cpu"))

        row = {"case_id": cid, "mode": mode,
               "l1_whole": whole_l1, "psnr_whole": whole_psnr,
               "ssim_whole": whole_ssim}

        for comp in ("torso", "subcu"):
            m_np = masks[comp]
            mask = torch.from_numpy(m_np).unsqueeze(0).unsqueeze(0).float().to(p_t.device)
            if mask.sum() < 1:
                row[f"l1_{comp}"] = row[f"rho_{comp}"] = float("nan")
                row[f"bias_{comp}"] = row[f"loa_{comp}"] = float("nan")
            else:
                with torch.no_grad():
                    row[f"l1_{comp}"] = masked_l1(p_t, r_t, mask).item()
                sel = m_np > 0
                pm, rm = pred[sel].flatten(), gt[sel].flatten()
                if pm.size > 1:
                    row[f"rho_{comp}"]  = float(np.corrcoef(pm, rm)[0, 1])
                    diffs = pm - rm
                    row[f"bias_{comp}"] = float(np.mean(diffs))
                    row[f"loa_{comp}"]  = float(1.96 * np.std(diffs))
                else:
                    row[f"rho_{comp}"] = row[f"bias_{comp}"] = row[f"loa_{comp}"] = float("nan")
            del mask

        rows.append(row)
        del p_t, r_t
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not rows:
        return {"mode": mode, "n_cases": 0}, []

    csv_path = os.path.join(output_dir, f"metrics_{mode}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    summary = {"mode": mode, "n_cases": len(rows)}
    for key in rows[0]:
        if key in ("case_id", "mode"):
            continue
        vals = [r[key] for r in rows
                if isinstance(r[key], float) and not np.isnan(r[key])]
        if vals:
            summary[f"{key}_mean"] = float(np.mean(vals))
            summary[f"{key}_std"]  = float(np.std(vals))

    return summary, rows

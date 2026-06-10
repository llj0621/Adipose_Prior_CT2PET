"""Main entry point: M0 → M1 → M2 → M2L → M3 sequential training + evaluation.

Usage
-----
New experiment::

    python train.py

Resume from an existing experiment directory::

    python train.py /path/to/results/1
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS",       "1")
os.environ.setdefault("MKL_NUM_THREADS",       "1")

import sys
import csv
import json
import random
import numpy as np
import torch

from adipose_ct2pet.dataset      import PatchDataset, build_volume_cache
from adipose_ct2pet.models.generator   import UNet3DGenerator
from adipose_ct2pet.models.refine_net  import RefinementUNet3D
from adipose_ct2pet.train_stage1 import train_stage1
from adipose_ct2pet.train_stage2 import train_stage2, _MODE_IN_CH
from adipose_ct2pet.evaluate     import evaluate
from adipose_ct2pet.utils        import next_exp_dir
import adipose_ct2pet.config as cfg


# ── Helpers ───────────────────────────────────────────────────

def discover_cases():
    """Return case IDs that have CT + PET + both fat masks."""
    cases = []
    for d in sorted(os.listdir(cfg.DATA_DIR)):
        if not os.path.isdir(os.path.join(cfg.DATA_DIR, d)):
            continue
        needed = [
            os.path.join(cfg.DATA_DIR, d, "CT.nii.gz"),
            os.path.join(cfg.DATA_DIR, d, "PET.nii.gz"),
            os.path.join(cfg.DATA_DIR, d, "torso_fat.nii.gz"),
            os.path.join(cfg.DATA_DIR, d, "subcutaneous_fat.nii.gz"),
        ]
        if all(os.path.exists(p) for p in needed):
            cases.append(d)
    return cases


def split_cases(cases, seed=cfg.RANDOM_SEED):
    rng = random.Random(seed)
    cases = sorted(cases)
    rng.shuffle(cases)
    n = len(cases)
    n_train = int(n * cfg.SPLIT[0])
    n_val   = int(n * cfg.SPLIT[1])
    return cases[:n_train], cases[n_train:n_train + n_val], cases[n_train + n_val:]


def make_dataset(cases, load_masks):
    return PatchDataset(
        cfg.DATA_DIR, cases,
        target_xy=cfg.TARGET_XY, patch_z=cfg.PATCH_Z,
        patch_stride=cfg.PATCH_STRIDE,
        ct_clip=cfg.CT_CLIP, pet_clip=cfg.PET_CLIP,
        load_masks=load_masks,
    )


def summary_from_csv(csv_path, mode):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"mode": mode, "n_cases": 0}
    summary = {"mode": mode, "n_cases": len(rows)}
    for key in rows[0]:
        if key in ("case_id", "mode"):
            continue
        vals = []
        for r in rows:
            try:
                v = float(r[key])
                if not np.isnan(v):
                    vals.append(v)
            except (ValueError, TypeError):
                pass
        if vals:
            summary[f"{key}_mean"] = float(np.mean(vals))
            summary[f"{key}_std"]  = float(np.std(vals))
    return summary


def save_summary(exp_dir, rows):
    order = ["M0", "M1", "M2", "M2L", "M3"]
    rows  = sorted(rows, key=lambda s: order.index(s["mode"]) if s["mode"] in order else 99)
    with open(os.path.join(exp_dir, "summary.json"), "w") as f:
        json.dump(rows, f, indent=2)


def print_summary(rows):
    order = ["M0", "M1", "M2", "M2L", "M3"]
    rows  = sorted(rows, key=lambda s: order.index(s["mode"]) if s["mode"] in order else 99)
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Mode':<6} {'L1_all':<8} {'L1_VAT':<8} {'L1_SAT':<8}"
          f" {'PSNR':<8} {'SSIM':<8} {'ρ_VAT':<8} {'ρ_SAT':<8}")
    print("-" * 70)
    for s in rows:
        def v(k):
            val = s.get(k, float("nan"))
            return f"{val:.4f}" if isinstance(val, float) else str(val)
        print(f"{s['mode']:<6} {v('l1_whole_mean'):<8} {v('l1_torso_mean'):<8}"
              f" {v('l1_subcu_mean'):<8} {v('psnr_whole_mean'):<8}"
              f" {v('ssim_whole_mean'):<8} {v('rho_torso_mean'):<8}"
              f" {v('rho_subcu_mean'):<8}")


# ── Main ──────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # Experiment directory
    if len(sys.argv) >= 2:
        exp_dir = sys.argv[1]
        if not os.path.isdir(exp_dir):
            sys.exit(f"[Error] {exp_dir} does not exist")
        resume = True
    else:
        exp_dir = next_exp_dir(cfg.RESULTS_ROOT)
        resume  = False
    print(f"[Exp] {exp_dir} ({'resume' if resume else 'new'})")

    # Data split
    split_path = os.path.join(exp_dir, "split.json")
    if resume and os.path.exists(split_path):
        with open(split_path) as f:
            sp = json.load(f)
        train_cases, val_cases, test_cases = sp["train"], sp["val"], sp["test"]
    else:
        all_cases = discover_cases()
        if cfg.NUM_CASES is not None:
            rng = random.Random(cfg.RANDOM_SEED)
            rng.shuffle(all_cases)
            all_cases = all_cases[:cfg.NUM_CASES]
        train_cases, val_cases, test_cases = split_cases(all_cases)
        with open(split_path, "w") as f:
            json.dump({"train": train_cases, "val": val_cases,
                       "test": test_cases}, f)
    print(f"[Split] Train={len(train_cases)} Val={len(val_cases)} Test={len(test_cases)}")

    summary_rows = []
    if resume and os.path.exists(os.path.join(exp_dir, "summary.json")):
        with open(os.path.join(exp_dir, "summary.json")) as f:
            summary_rows = json.load(f)

    # ── Stage 1 ───────────────────────────────────────────────
    g_ckpt = os.path.join(exp_dir, "checkpoints", "G_best.pth")
    val_s1 = build_volume_cache(make_dataset(val_cases, load_masks=False))

    if os.path.exists(g_ckpt):
        print("[S1] G_best.pth found — skipping training")
        G = UNet3DGenerator(in_ch=1, out_ch=1, base=cfg.G_BASE).to(device)
        G.load_state_dict(torch.load(g_ckpt, map_location=device))
        G.eval()
    else:
        print("\n" + "=" * 60 + "\nStage 1: M0 baseline\n" + "=" * 60)
        _, G = train_stage1(make_dataset(train_cases, False),
                            make_dataset(val_cases,   False),
                            exp_dir, device, val_cache=val_s1)
    del val_s1

    # M0 evaluation
    out_m0   = os.path.join(exp_dir, "outputs_M0")
    csv_m0   = os.path.join(out_m0, "metrics_M0.csv")
    test_c   = build_volume_cache(make_dataset(test_cases, load_masks=True))
    if os.path.exists(csv_m0):
        m0_sum = summary_from_csv(csv_m0, "M0")
    else:
        m0_sum, _ = evaluate(G, None, "M0", test_cases, device, out_m0, cache=test_c)
    summary_rows = [s for s in summary_rows if s.get("mode") != "M0"] + [m0_sum]
    save_summary(exp_dir, summary_rows)
    del test_c

    # ── Stage 2 (M1 / M2 / M2L / M3) ─────────────────────────
    for mode in ("M1", "M2", "M2L", "M3"):
        needs_mask = mode in ("M2", "M2L", "M3")
        r_ckpt     = os.path.join(exp_dir, "checkpoints", f"R_{mode}_best.pth")
        val_m      = build_volume_cache(make_dataset(val_cases, needs_mask))

        if os.path.exists(r_ckpt):
            print(f"[S2-{mode}] R_{mode}_best.pth found — skipping training")
            R = RefinementUNet3D(in_ch=_MODE_IN_CH[mode], base_ch=cfg.R_BASE).to(device)
            R.load_state_dict(torch.load(r_ckpt, map_location=device))
            R.eval()
        else:
            print(f"\n{'='*60}\nStage 2: {mode}\n{'='*60}")
            _, R = train_stage2(G, mode,
                                make_dataset(train_cases, needs_mask),
                                make_dataset(val_cases,   needs_mask),
                                exp_dir, device, val_cache=val_m)
        del val_m

        out_m    = os.path.join(exp_dir, f"outputs_{mode}")
        csv_m    = os.path.join(out_m, f"metrics_{mode}.csv")
        test_m   = build_volume_cache(make_dataset(test_cases, True))
        if os.path.exists(csv_m):
            m_sum = summary_from_csv(csv_m, mode)
        else:
            m_sum, _ = evaluate(G, R, mode, test_cases, device, out_m, cache=test_m)
        del test_m

        summary_rows = [s for s in summary_rows if s.get("mode") != mode] + [m_sum]
        save_summary(exp_dir, summary_rows)

    print(f"\n[Done] Summary: {os.path.join(exp_dir, 'summary.json')}")
    print_summary(summary_rows)


if __name__ == "__main__":
    main()

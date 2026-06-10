"""Stage 1: Train UNet3DGenerator + MultiScalePatchGAN3D (pix2pix baseline)."""

import os
import csv
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from adipose_ct2pet.models.generator     import UNet3DGenerator
from adipose_ct2pet.models.discriminator import MultiScalePatchGAN3D
from adipose_ct2pet.dataset  import cached_patches
from adipose_ct2pet.losses   import d_loss, g_adv_loss, l1, feature_matching_loss
from adipose_ct2pet.ms_ssim  import ms_ssim_loss
from adipose_ct2pet.utils    import denorm_pet, gaussian_weights, ensure_dir, amp_ok
from skimage.metrics import peak_signal_noise_ratio as psnr
import adipose_ct2pet.config as cfg


def build_models(device):
    G = UNet3DGenerator(in_ch=1, out_ch=1, base=cfg.G_BASE).to(device)
    D = MultiScalePatchGAN3D(
        in_ch=1, out_ch=1, base_ch=cfg.D_BASE,
        num_scales=cfg.NUM_SCALES, min_spatial=cfg.MIN_SPATIAL,
    ).to(device)
    return G, D


@torch.no_grad()
def validate(G, dataset, device, n_cases=None, cache=None):
    G.eval()
    ids     = (dataset.case_ids[:n_cases] if n_cases else dataset.case_ids)
    weights = gaussian_weights(cfg.PATCH_Z).to(device)
    totals  = {"l1": 0.0, "psnr": 0.0, "ssim": 0.0, "n": 0}

    for cid in ids:
        z_all   = cache[cid]["ct"].shape[0] if cache else dataset._load_case(cid)[0].shape[0]
        vol_sum = torch.zeros(z_all, cfg.TARGET_XY, cfg.TARGET_XY, device=device)
        wgt_sum = torch.zeros_like(vol_sum)
        vol_real = torch.zeros_like(vol_sum)

        patches = (cached_patches(cache[cid], cfg.PATCH_Z, cfg.PATCH_STRIDE)
                   if cache else dataset.all_patches(cid))

        for z0, batch in patches:
            ct   = batch["ct"].unsqueeze(0).to(device)
            pet  = batch["pet"].unsqueeze(0).to(device)
            fake = G(ct).float()
            w    = weights.view(1, -1, 1, 1)
            vol_sum[z0:z0 + cfg.PATCH_Z]  += fake[0, 0] * w[0]
            wgt_sum[z0:z0 + cfg.PATCH_Z]  += w[0]
            vol_real[z0:z0 + cfg.PATCH_Z]  = pet[0, 0]

        fused   = vol_sum / (wgt_sum + 1e-8)
        fake_np = denorm_pet(fused.cpu().numpy(), cfg.PET_CLIP[1])
        real_np = denorm_pet(vol_real.cpu().numpy(), cfg.PET_CLIP[1])

        totals["l1"]   += torch.mean(torch.abs(fused - vol_real)).item()
        totals["psnr"] += psnr(real_np, fake_np, data_range=cfg.PET_CLIP[1])
        totals["ssim"] += 1.0 - ms_ssim_loss(
            fused.unsqueeze(0).unsqueeze(0).float(),
            vol_real.unsqueeze(0).unsqueeze(0).float()).item()
        totals["n"]    += 1

    G.train()
    n = totals["n"]
    return totals["l1"] / n, totals["psnr"] / n, totals["ssim"] / n


def train_stage1(train_ds, val_ds, exp_dir, device, val_cache=None):
    """Train G + D.  Returns (checkpoint_path, trained_G)."""
    G, D       = build_models(device)
    opt_G      = torch.optim.Adam(G.parameters(), lr=cfg.S1_LR, betas=cfg.S1_BETAS)
    opt_D      = torch.optim.Adam(D.parameters(), lr=cfg.S1_LR, betas=cfg.S1_BETAS)
    use_amp    = amp_ok(device, cfg.USE_AMP)
    scaler     = torch.amp.GradScaler("cuda", enabled=use_amp)
    lambda_fm  = float(getattr(cfg, "S1_LAMBDA_FM", 0.0))

    loader     = DataLoader(train_ds, batch_size=cfg.S1_BATCH_SIZE,
                            shuffle=True, num_workers=cfg.NUM_WORKERS,
                            pin_memory=True, drop_last=True)
    ckpt_dir   = os.path.join(exp_dir, "checkpoints")
    ensure_dir(ckpt_dir)
    log_path   = os.path.join(exp_dir, "stage1_log.csv")

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "g_loss", "d_loss",
                                 "val_l1", "val_psnr", "val_ssim"])

    best_ssim, best_epoch = -1.0, 0

    for epoch in range(1, cfg.S1_EPOCHS + 1):
        G.train(); D.train()
        sum_g = sum_d = n_iter = 0

        pbar = tqdm(loader, desc=f"S1 E{epoch:03d}", leave=False)
        for batch in pbar:
            ct  = batch["ct"].to(device)
            pet = batch["pet"].to(device)

            # ── D step ──────────────────────────────────────
            for p in D.parameters():
                p.requires_grad = True
            opt_D.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                with torch.no_grad():
                    fake = G(ct)
                loss_D = d_loss(D(ct, pet), D(ct, fake.detach()))
            scaler.scale(loss_D).backward()
            scaler.step(opt_D); scaler.update()

            # ── G step ──────────────────────────────────────
            for p in D.parameters():
                p.requires_grad = False
            opt_G.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                fake = G(ct)
                if lambda_fm > 0:
                    d_fake_out, fake_feats = D(ct, fake, return_feats=True)
                    with torch.no_grad():
                        _, real_feats = D(ct, pet, return_feats=True)
                    loss_fm = sum(feature_matching_loss(rf, ff)
                                  for rf, ff in zip(real_feats, fake_feats)) \
                              / max(1, len(fake_feats))
                else:
                    d_fake_out = D(ct, fake)
                    loss_fm    = torch.zeros((), device=device)
                loss_G = (g_adv_loss(d_fake_out)
                          + cfg.S1_LAMBDA_L1 * l1(fake, pet)
                          + lambda_fm * loss_fm)
            scaler.scale(loss_G).backward()
            scaler.step(opt_G); scaler.update()

            sum_g += loss_G.item(); sum_d += loss_D.item(); n_iter += 1
            pbar.set_postfix(G=f"{loss_G.item():.3f}", D=f"{loss_D.item():.3f}")

        for p in D.parameters():
            p.requires_grad = True
        pbar.close()

        v_l1, v_psnr, v_ssim = validate(
            G, val_ds, device, n_cases=cfg.S1_VAL_N_CASES, cache=val_cache)
        tqdm.write(f"[S1 E{epoch:03d}] G={sum_g/n_iter:.4f} D={sum_d/n_iter:.4f}"
                   f"  Val L1={v_l1:.4f} PSNR={v_psnr:.2f} SSIM={v_ssim:.4f}")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, sum_g / n_iter, sum_d / n_iter, v_l1, v_psnr, v_ssim])

        if v_ssim > best_ssim:
            best_ssim, best_epoch = v_ssim, epoch
            torch.save(G.state_dict(), os.path.join(ckpt_dir, "G_best.pth"))

    best_path = os.path.join(ckpt_dir, "G_best.pth")
    if os.path.exists(best_path):
        G.load_state_dict(torch.load(best_path, map_location=device))
        tqdm.write(f"[S1] Best G loaded (epoch {best_epoch}, SSIM={best_ssim:.4f})")
    else:
        torch.save(G.state_dict(), best_path)

    return best_path, G

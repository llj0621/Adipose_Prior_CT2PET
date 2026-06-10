"""Stage 2: Train RefinementUNet3D with optional adipose-prior conditioning."""

import os
import csv
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from adipose_ct2pet.models.refine_net import RefinementUNet3D
from adipose_ct2pet.dataset  import cached_patches
from adipose_ct2pet.losses   import l1, masked_l1, fat_stratified_losses, get_fat_weights
from adipose_ct2pet.ms_ssim  import ms_ssim_loss
from adipose_ct2pet.utils    import denorm_pet, gaussian_weights, ensure_dir, amp_ok
from skimage.metrics import peak_signal_noise_ratio as psnr
import adipose_ct2pet.config as cfg

# Mode → RefinementUNet input channels
_MODE_IN_CH = {"M1": 2, "M2L": 2, "M2": 4, "M3": 4}


def _uses_mask_input(mode):
    return mode in ("M2", "M3")


def _uses_mask_loss(mode):
    return mode in ("M2L", "M3")


def build_refine_net(mode, device):
    return RefinementUNet3D(in_ch=_MODE_IN_CH[mode], base_ch=cfg.R_BASE).to(device)


@torch.no_grad()
def validate(G, R, mode, dataset, device, n_cases=None, cache=None):
    G.eval(); R.eval()
    ids     = (dataset.case_ids[:n_cases] if n_cases else dataset.case_ids)
    weights = gaussian_weights(cfg.PATCH_Z).to(device)
    track   = _uses_mask_input(mode) or _uses_mask_loss(mode)
    tot     = {"l1": 0.0, "psnr": 0.0, "ssim": 0.0,
               "torso_l1": 0.0, "subcu_l1": 0.0, "n": 0}

    for cid in ids:
        z_all    = cache[cid]["ct"].shape[0] if cache else dataset._load_case(cid)[0].shape[0]
        vol_sum  = torch.zeros(z_all, cfg.TARGET_XY, cfg.TARGET_XY, device=device)
        wgt_sum  = torch.zeros_like(vol_sum)
        vol_real = torch.zeros_like(vol_sum)
        vol_t    = torch.zeros_like(vol_sum)
        vol_s    = torch.zeros_like(vol_sum)

        patches = (cached_patches(cache[cid], cfg.PATCH_Z, cfg.PATCH_STRIDE)
                   if cache else dataset.all_patches(cid))

        for z0, batch in patches:
            ct   = batch["ct"].unsqueeze(0).to(device)
            pet  = batch["pet"].unsqueeze(0).to(device)
            fake = G(ct).float()

            if _uses_mask_input(mode):
                refined = R(fake, ct,
                            batch["torso_mask"].unsqueeze(0).to(device),
                            batch["subcu_mask"].unsqueeze(0).to(device))
            else:
                refined = R(fake, ct)

            w   = weights.view(1, -1, 1, 1)
            slc = slice(z0, z0 + cfg.PATCH_Z)
            vol_sum[slc]  += refined[0, 0] * w[0]
            wgt_sum[slc]  += w[0]
            vol_real[slc]  = pet[0, 0]
            if track:
                vol_t[slc] = torch.maximum(
                    vol_t[slc], batch["torso_mask"].squeeze(0).to(device))
                vol_s[slc] = torch.maximum(
                    vol_s[slc], batch["subcu_mask"].squeeze(0).to(device))

        fused   = vol_sum / (wgt_sum + 1e-8)
        fake_np = denorm_pet(fused.cpu().numpy(), cfg.PET_CLIP[1])
        real_np = denorm_pet(vol_real.cpu().numpy(), cfg.PET_CLIP[1])

        tot["l1"]   += torch.mean(torch.abs(fused - vol_real)).item()
        tot["psnr"] += psnr(real_np, fake_np, data_range=cfg.PET_CLIP[1])
        tot["ssim"] += 1.0 - ms_ssim_loss(
            fused.unsqueeze(0).unsqueeze(0).float(),
            vol_real.unsqueeze(0).unsqueeze(0).float()).item()
        if track:
            tot["torso_l1"] += masked_l1(
                fused.unsqueeze(0).unsqueeze(0),
                vol_real.unsqueeze(0).unsqueeze(0),
                vol_t.unsqueeze(0).unsqueeze(0)).item()
            tot["subcu_l1"] += masked_l1(
                fused.unsqueeze(0).unsqueeze(0),
                vol_real.unsqueeze(0).unsqueeze(0),
                vol_s.unsqueeze(0).unsqueeze(0)).item()
        tot["n"] += 1

    R.train()
    n = tot["n"]
    return (tot["l1"] / n, tot["psnr"] / n, tot["ssim"] / n,
            tot["torso_l1"] / n, tot["subcu_l1"] / n)


def train_stage2(G, mode, train_ds, val_ds, exp_dir, device, val_cache=None):
    """Freeze G and train RefineNet for one mode.  Returns (ckpt_path, R)."""
    G.eval()
    for p in G.parameters():
        p.requires_grad = False

    R        = build_refine_net(mode, device)
    opt_R    = torch.optim.Adam(R.parameters(), lr=cfg.S2_LR, betas=cfg.S2_BETAS)
    use_amp  = amp_ok(device, cfg.USE_AMP) and getattr(cfg, "S2_USE_AMP", False)
    scaler   = torch.amp.GradScaler("cuda", enabled=use_amp)
    grad_clip = float(getattr(cfg, "S2_GRAD_CLIP", 0.0))

    loader   = DataLoader(train_ds, batch_size=cfg.S2_BATCH_SIZE,
                          shuffle=True, num_workers=cfg.NUM_WORKERS, pin_memory=True)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    ensure_dir(ckpt_dir)
    log_path = os.path.join(exp_dir, f"stage2_{mode}_log.csv")

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "loss", "val_l1", "val_psnr", "val_ssim",
                                 "val_torso_l1", "val_subcu_l1"])

    composite = mode in ("M2L", "M3")
    alpha     = float(getattr(cfg, "S2_BEST_COMPOSITE_ALPHA", 0.1))
    best_score, best_epoch = -float("inf"), 0

    for epoch in range(1, cfg.S2_EPOCHS + 1):
        R.train()
        sum_loss = n_iter = 0

        w_t = w_s = None
        if _uses_mask_loss(mode):
            w_t, w_s = get_fat_weights(
                epoch, cfg.S2_EPOCHS,
                cfg.FAT_W_TORSO, cfg.FAT_W_SUBCU, cfg.FAT_MAX_SCALE)

        pbar = tqdm(loader, desc=f"S2-{mode} E{epoch:02d}", leave=False)
        for batch in pbar:
            ct    = batch["ct"].to(device)
            pet   = batch["pet"].to(device)
            torso = batch.get("torso_mask")
            subcu = batch.get("subcu_mask")
            if torso is not None:
                torso = torso.to(device)
                subcu = subcu.to(device)

            opt_R.zero_grad()
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=use_amp):
                    fake = G(ct).float()

            with torch.amp.autocast("cuda", enabled=use_amp):
                refined = R(fake, ct, torso, subcu) if _uses_mask_input(mode) \
                          else R(fake, ct)
                loss = cfg.S2_LAMBDA_L1 * l1(refined, pet)
                if _uses_mask_loss(mode):
                    fl   = fat_stratified_losses(refined, pet, torso, subcu)
                    loss = loss + w_t * fl["torso"] + w_s * fl["subcu"]

            # MS-SSIM outside autocast — fp16 multi-scale convs are numerically unstable
            loss = loss + cfg.S2_LAMBDA_MS * ms_ssim_loss(refined.float(), pet.float())

            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(opt_R)
                torch.nn.utils.clip_grad_norm_(R.parameters(), grad_clip)
            scaler.step(opt_R); scaler.update()

            sum_loss += loss.item(); n_iter += 1
            pbar.set_postfix(L=f"{loss.item():.4f}")

        pbar.close()

        v_l1, v_psnr, v_ssim, v_t, v_s = validate(
            G, R, mode, val_ds, device, n_cases=cfg.S2_VAL_N_CASES, cache=val_cache)
        score = (v_ssim - alpha * (v_t + v_s) / 2.0) if composite else v_ssim
        tqdm.write(
            f"[S2-{mode} E{epoch:02d}] Loss={sum_loss/n_iter:.4f}"
            f"  Val L1={v_l1:.4f} PSNR={v_psnr:.2f} SSIM={v_ssim:.4f}"
            f"  VAT={v_t:.4f} SAT={v_s:.4f}"
            + (f"  Score={score:.4f}" if composite else ""))

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, sum_loss / n_iter, v_l1, v_psnr, v_ssim, v_t, v_s])

        if epoch % cfg.SAVE_SAMPLES_EVERY == 0:
            torch.save(R.state_dict(),
                       os.path.join(ckpt_dir, f"R_{mode}_e{epoch:02d}.pth"))

        if score > best_score:
            best_score, best_epoch = score, epoch
            torch.save(R.state_dict(), os.path.join(ckpt_dir, f"R_{mode}_best.pth"))

    best_path = os.path.join(ckpt_dir, f"R_{mode}_best.pth")
    R.load_state_dict(torch.load(best_path, map_location=device))
    tqdm.write(f"[S2-{mode}] Best RefineNet (epoch {best_epoch}, score={best_score:.4f})")
    return best_path, R

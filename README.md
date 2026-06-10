# Adipose-Prior CT-to-PET Synthesis

A two-stage 3D GAN framework for synthesizing whole-body PET images from CT, with adipose-tissue-aware refinement to improve visceral (VAT) and subcutaneous (SAT) fat quantification accuracy.

---

## Overview

Standard pix2pix-based CT-to-PET synthesis treats all body regions equally, leading to systematic errors in metabolically active adipose compartments. This work introduces a lightweight refinement stage that conditions on segmented fat masks to correct those errors without retraining the full GAN.

**Two-stage pipeline:**

```
CT ──► [Stage 1: UNet3D GAN] ──► coarse PET
                                       │
       fat masks (VAT, SAT) ──────────►│
                                       ▼
                          [Stage 2: Refinement UNet] ──► refined PET
```

**Four refinement modes** allow systematic ablation of where the adipose prior is applied:

| Mode | Mask as input | Mask in loss | Description |
|------|:---:|:---:|-------------|
| M0   | —   | —   | Stage 1 only (pix2pix baseline) |
| M1   | ✗   | ✗   | Refinement, no adipose prior |
| M2L  | ✗   | ✓   | Adipose prior in loss only |
| M2   | ✓   | ✗   | Adipose prior as input only |
| M3   | ✓   | ✓   | Full adipose prior (input + loss) |

---

## Architecture

### Stage 1 — `UNet3DGenerator`

4-level 3D UNet encoder–decoder with skip connections.  
- Encoder: strided Conv3d + InstanceNorm + LeakyReLU  
- Decoder: trilinear upsample + Conv3d (avoids ConvTranspose checkerboard artifacts)  
- Bottleneck: single Conv3d residual block  
- Output: tanh → [-1, 1] (maps to SUV space during inference)

### Stage 1 — `MultiScalePatchGAN3D`

Two-scale 3D PatchGAN discriminator with Spectral Normalization.  
- Full-resolution + 2× average-pooled inputs evaluated independently  
- Spectral Norm on all conv layers for training stability (SN-GAN, Miyato et al. 2018)

### Stage 2 — `RefinementUNet3D`

Lightweight 3-level UNet for **residual refinement** on frozen Stage-1 output.  
- Output = Stage-1 fake + learned residual  
- Zero-initialized output conv → identity map at step 0  
- Accepts 2 channels (M1/M2L) or 4 channels (M2/M3) depending on mode

---

## Installation

```bash
git clone https://github.com/<your-username>/Adipose_Prior_CT2PET.git
cd Adipose_Prior_CT2PET
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.9, PyTorch ≥ 2.0, CUDA recommended.

---

## Data Preparation

Expected directory layout:

```
DATA_DIR/
  {case_id}/
    CT.nii.gz
    PET.nii.gz
    torso_fat.nii.gz          # visceral (VAT) binary mask
    subcutaneous_fat.nii.gz   # subcutaneous (SAT) binary mask
```

Fat masks can be generated with any body-composition segmentation tool (e.g., TotalSegmentator, nnU-Net).

---

## Configuration

Edit [`adipose_ct2pet/config.py`](adipose_ct2pet/config.py) before training:

```python
DATA_DIR     = "/path/to/data"
RESULTS_ROOT = "/path/to/results"
```

Key hyperparameters (with defaults):

| Parameter | Default | Notes |
|-----------|---------|-------|
| `TARGET_XY` | 256 | XY resize target (native 512 → 256) |
| `PATCH_Z` | 32 | Z-axis patch depth |
| `S1_EPOCHS` | 80 | Stage 1 training epochs |
| `S2_EPOCHS` | 30 | Stage 2 training epochs |
| `FAT_W_TORSO` | 3.0 | VAT loss weight in M3 |
| `FAT_W_SUBCU` | 2.0 | SAT loss weight in M3 |

---

## Training

Run the full pipeline (M0 → M1 → M2 → M2L → M3) sequentially:

```bash
python train.py
```

Resume an interrupted experiment:

```bash
python train.py /path/to/results/1
```

The script auto-detects existing checkpoints and skips completed stages. Training outputs:

```
results/1/
  split.json                  # reproducible train/val/test split
  checkpoints/
    G_best.pth
    R_M1_best.pth
    R_M2_best.pth
    R_M2L_best.pth
    R_M3_best.pth
  stage1_log.csv
  stage2_M3_log.csv
  outputs_M3/
    {case_id}_PET_M3.nii.gz
    metrics_M3.csv
  summary.json
```

---

## Inference

```python
import torch
from adipose_ct2pet.models.generator  import UNet3DGenerator
from adipose_ct2pet.models.refine_net import RefinementUNet3D
from adipose_ct2pet.inference import infer
import adipose_ct2pet.config as cfg

device = torch.device("cuda")

G = UNet3DGenerator(in_ch=1, out_ch=1, base=cfg.G_BASE).to(device)
G.load_state_dict(torch.load("results/1/checkpoints/G_best.pth"))

R = RefinementUNet3D(in_ch=4, base_ch=cfg.R_BASE).to(device)  # in_ch=4 for M3
R.load_state_dict(torch.load("results/1/checkpoints/R_M3_best.pth"))

pet_suv, affine = infer(G, R, mode="M3", case_id="case_001",
                         device=device, output_dir="outputs/", save_nifti=True)
# pet_suv: np.ndarray [Z, H, W] in SUV units
```

---

## Evaluation Metrics

Per-case metrics written to `metrics_{mode}.csv`:

| Metric | Description |
|--------|-------------|
| `l1_whole` | Mean absolute error (SUV) — whole body |
| `psnr_whole` | Peak signal-to-noise ratio |
| `ssim_whole` | Multi-scale SSIM (3D) |
| `l1_torso` | MAE restricted to VAT region |
| `l1_subcu` | MAE restricted to SAT region |
| `rho_torso` | Pearson ρ within VAT |
| `rho_subcu` | Pearson ρ within SAT |
| `bias_torso` | Mean bias (Bland–Altman) within VAT |
| `loa_torso` | 1.96 × SD (limits of agreement) within VAT |

---

## Loss Functions

| Loss | Stage | Description |
|------|-------|-------------|
| BCE adversarial | S1 | Multi-scale PatchGAN |
| L1 reconstruction | S1 | Global voxel-level fidelity |
| Feature matching | S1 | Optional; matches D intermediate features |
| L1 reconstruction | S2 | Global voxel-level fidelity |
| MS-SSIM | S2 | 3-level multi-scale structural similarity |
| VAT masked L1 | S2 (M2L/M3) | L1 restricted to visceral fat region |
| SAT masked L1 | S2 (M2L/M3) | L1 restricted to subcutaneous fat region |

---

## Repository Structure

```
Adipose_Prior_CT2PET/
├── adipose_ct2pet/
│   ├── __init__.py
│   ├── config.py               # all hyperparameters and data paths
│   ├── dataset.py              # PatchDataset, volume cache, sliding-window
│   ├── losses.py               # adversarial, L1, feature matching, fat-stratified
│   ├── ms_ssim.py              # 3D Multi-Scale SSIM
│   ├── utils.py                # normalisation, blending weights, helpers
│   ├── inference.py            # sliding-window inference + NIfTI export
│   ├── train_stage1.py         # GAN training loop
│   ├── train_stage2.py         # refinement training loop
│   ├── evaluate.py             # test-set evaluation + metric export
│   └── models/
│       ├── __init__.py
│       ├── generator.py        # UNet3DGenerator
│       ├── discriminator.py    # MultiScalePatchGAN3D
│       └── refine_net.py       # RefinementUNet3D
├── train.py                    # main entry point
├── requirements.txt
└── setup.py
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{adipose_ct2pet_2025,
  author    = {<Author>},
  title     = {Adipose-Prior CT-to-PET Synthesis},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/<your-username>/Adipose_Prior_CT2PET}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

"""Configuration for the two-stage CT-to-PET synthesis pipeline."""

# ── Data paths (replace with your own) ───────────────────────
# Expected layout:
#   DATA_DIR/{case_id}/CT.nii.gz
#   DATA_DIR/{case_id}/PET.nii.gz
#   DATA_DIR/{case_id}/torso_fat.nii.gz
#   DATA_DIR/{case_id}/subcutaneous_fat.nii.gz
DATA_DIR     = "/path/to/data"
RESULTS_ROOT = "/path/to/results"

# ── Subset control ────────────────────────────────────────────
NUM_CASES = None          # None = all; int = random subsample

# ── Preprocessing ─────────────────────────────────────────────
TARGET_XY    = 256        # resize native 512 → 256
PATCH_Z      = 32         # Z-axis patch depth
PATCH_STRIDE = 16         # 50% overlap
CT_CLIP      = (-1000, 1000)
PET_CLIP     = (0, 5)     # SUV range

# ── Dataset split ─────────────────────────────────────────────
SPLIT       = (0.8, 0.1, 0.1)
RANDOM_SEED = 42

# ── Architecture ──────────────────────────────────────────────
G_BASE      = 32          # UNet3DGenerator base channels
D_BASE      = 32          # PatchGAN base channels
R_BASE      = 32          # RefinementUNet base channels
NUM_SCALES  = 2           # multi-scale discriminator count
MIN_SPATIAL = 16          # minimum spatial dim for discriminator

# ── Stage 1 (GAN baseline) ────────────────────────────────────
S1_BATCH_SIZE    = 4
S1_LR            = 2e-4
S1_BETAS         = (0.5, 0.999)
S1_EPOCHS        = 80
S1_LAMBDA_L1     = 100
S1_LAMBDA_FM     = 0      # feature-matching weight; 0 to disable
S1_VAL_N_CASES   = 20

# ── Stage 2 (refinement) ──────────────────────────────────────
S2_BATCH_SIZE          = 1
S2_LR                  = 2e-5
S2_BETAS               = (0.5, 0.999)
S2_EPOCHS              = 30
S2_LAMBDA_L1           = 1.0
S2_LAMBDA_MS           = 0.2
S2_USE_AMP             = False   # MS-SSIM is fp16-unstable; keep False
S2_GRAD_CLIP           = 1.0
S2_VAL_N_CASES         = 20
S2_BEST_COMPOSITE_ALPHA = 0.1   # fat-region weight in best-model score

# ── Adipose compartment weights (M3) ─────────────────────────
# max_scale=1.0 → static weights (no linear ramp)
FAT_MAX_SCALE = 1.0
FAT_W_TORSO   = 3.0      # VAT (visceral) — higher clinical priority
FAT_W_SUBCU   = 2.0      # SAT (subcutaneous)

# ── Mixed precision ───────────────────────────────────────────
USE_AMP = True

# ── Misc ──────────────────────────────────────────────────────
NUM_WORKERS        = 8
SAVE_SAMPLES_EVERY = 10

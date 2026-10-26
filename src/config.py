"""Global configuration shared by every script.

Paths can be redirected with environment variables, e.g. on Colab:
    DL07_WORKDIR=/content/drive/MyDrive/dl07   (checkpoints + results)
    DL07_DATA=/content/data                    (dataset cache)
Set DL07_QUICK=1 for a fast smoke test (tiny subset, 1 epoch, few FID samples).
"""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKDIR = os.environ.get("DL07_WORKDIR", ROOT)
DATA_DIR = os.environ.get("DL07_DATA", os.path.join(ROOT, "data"))
CKPT_DIR = os.path.join(WORKDIR, "checkpoints")
RESULTS_DIR = os.path.join(WORKDIR, "results")
LOG_DIR = os.path.join(RESULTS_DIR, "logs")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
SAMPLE_DIR = os.path.join(RESULTS_DIR, "samples")

QUICK = os.environ.get("DL07_QUICK", "0") == "1"

# --- standardized evaluation protocol (Bagian 3 panduan) ---
SEED = 42
DATASET = "FashionMNIST"
SPLIT = (0.8, 0.1, 0.1)          # train / val / test over all 70,000 images
BATCH_SIZE = 128
IMG_SHAPE = (1, 28, 28)

# --- architecture ---
HIDDEN = 256                     # 784 -> 256 -> 64 shared by MLP classifier and AE family
LATENT_AE = 64                   # Vanilla AE, Sparse AE, VAE, z_MLP
LATENT_GAN = 100                 # Vanilla GAN, DCGAN prior noise
SPARSE_LAMBDA = 1e-3             # weight of L1 penalty on the 256-d hidden code of Sparse AE
DDPM_T = 1000
DDPM_BETA = (1e-4, 0.02)         # linear schedule (Ho et al., 2020)
DDPM_BASE_CH = 64
EMA_DECAY = 0.999

# --- optimisation ---
EPOCHS = {
    "mlp": 30, "ae": 50, "sae": 50, "vae": 50,
    "gan": 100, "dcgan": 30, "ddpm": 40, "frozen_dec": 50,
}
LR = {
    "mlp": 1e-3, "ae": 1e-3, "sae": 1e-3, "vae": 1e-3,
    "gan": 2e-4, "dcgan": 2e-4, "ddpm": 2e-4, "frozen_dec": 1e-3,
}
GAN_BETAS = (0.5, 0.999)
MLP_TARGET_ACC = 0.88

# --- evaluation ---
FID_N = 5000                     # generated samples per model for FID (real side = full test set)
LATENCY_BATCH = 64
LATENCY_REPS = 50
LATENCY_WARMUP = 10
SNAPSHOT_EVERY = 10              # epochs between GAN sample snapshots / mode-coverage checks

if QUICK:
    EPOCHS = {k: 1 for k in EPOCHS}
    FID_N = 256
    LATENCY_REPS = 3
    LATENCY_WARMUP = 1
    SNAPSHOT_EVERY = 1
    DDPM_T = 50                  # smoke test only; real runs use T = 1000

for _d in (DATA_DIR, CKPT_DIR, RESULTS_DIR, LOG_DIR, FIG_DIR, SAMPLE_DIR):
    os.makedirs(_d, exist_ok=True)

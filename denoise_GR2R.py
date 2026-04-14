# ============================================================
# SEM Image Denoising — R2R / GR2R (pure PyTorch)
# ============================================================
# Based on:
#   [R2R]  "Recorrupted-to-Recorrupted: Unsupervised Deep Learning for Image Denoising"
#          Pang et al., CVPR 2021  (arXiv:2102.02234)
#          GitHub: github.com/PangTongyao/Recorrupted-to-Recorrupted-Unsupervised-
#                  Deep-Learning-for-Image-Denoising
#   [GR2R] "Generalized Recorrupted-to-Recorrupted: Self-Supervised Learning
#          Beyond Gaussian Noise"  Monroy, Bacca, Tachella, CVPR 2025
#          (arXiv:2412.04648)  GitHub: github.com/bemc22/GeneralizedR2R
#
# Differences from denoise_N2V.py:
#   + NO blind-spot masking — each patch is re-corrupted as an (y1, y2) pair;
#     MSE is applied to ALL pixels, giving a full-context receptive field.
#   + Gaussian re-corruption uses R2R asymmetric shared-ε pair (Pang 2021):
#       eps ~ N(0, σ_r²),  y1 = y + eps,  y2 = y − eps
#     Both views share the same ε; y2 = 2y − y1 is deterministic, giving
#     a lower-variance training target than two independent draws.
#   + Poisson re-corruption uses GR2R Binomial splitting (Monroy 2025):
#       z = round(y · photon_scale),  ω ~ Binomial(z, α_b)
#       y1 = (y − ω/photon_scale) / (1 − α_b),  y2 = y/α_b − (1−α_b)/α_b · y1
#     Preserves SEM shot-noise NEF structure; superior for low-count images.
#   + Noise std auto-estimated via Laplacian MAD (Immerkær 1996).
#   + --mc_samples > 1 enables Monte Carlo inference averaging (GR2R §5.3):
#     J independent y1 samples are passed through the model; outputs averaged.
#     Reduces prediction variance; recommended J=5–15 for best quality.
#   + --alpha scales Gaussian σ_r = α · σ_est (strength of re-corruption).
#   + --binomial_alpha sets Binomial success probability for --poisson mode
#     (recommended α_b = 0.15, Monroy et al.).
#
# Identical to denoise_N2V_multi.py:
#   = load_sem_image()  — ITU-R RGB→gray, float32 [0, 1]
#   = DoubleConvBlock, N2VUNet  — 4-level encoder-decoder, base_features=32
#   = _compute_padding(), predict_tiled()  — per-axis reflection padding (Opt 4),
#     Hann-window blending, batched GPU inference with infer_batch_size=8 (Opt 2)
#   = save_outputs()  — TIF in original range + 3-panel PNG (Original|Denoised|Diff×3)
#   = Adam + CosineAnnealingLR, lr=4e-4 → 1e-6
#
# Requirements: torch>=2.0.0  tifffile  matplotlib  numpy
# Usage:
#   python denoise_GR2R.py
#   python denoise_GR2R.py --input data/test_sem.tif --epochs 100
#   python denoise_GR2R.py --noise_std 0.05 --alpha 1.5
#   python denoise_GR2R.py --poisson --photon_scale 200 --binomial_alpha 0.15
#   python denoise_GR2R.py --mc_samples 5
# ============================================================

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import argparse
import time
from typing import List, Tuple

import numpy as np
import tifffile
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

torch.set_float32_matmul_precision('high')


# ============================================================
# 1. Image Loading
# ============================================================

def load_sem_image(path: str) -> Tuple[np.ndarray, float, float]:
    """Load SEM image, normalize to float32 [0, 1] grayscale.
    Returns (image, original_min, original_max) for later denormalization."""
    img = tifffile.imread(path).astype(np.float32)

    if img.ndim == 3 and img.shape[-1] == 3:
        img = img @ np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
    elif img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3] @ np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)

    img_min, img_max = float(img.min()), float(img.max())
    img = (img - img_min) / (img_max - img_min + 1e-8)
    return img.astype(np.float32), img_min, img_max


# ============================================================
# 2. UNet Architecture
# ============================================================

class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class N2VUNet(nn.Module):
    """4-level encoder-decoder UNet for Noise2Void (2D grayscale)."""

    def __init__(self, in_channels: int = 1, base_features: int = 32):
        super().__init__()
        f = base_features

        self.enc1 = DoubleConvBlock(in_channels, f)
        self.enc2 = DoubleConvBlock(f,     f * 2)
        self.enc3 = DoubleConvBlock(f * 2, f * 4)
        self.enc4 = DoubleConvBlock(f * 4, f * 8)
        self.pool = nn.MaxPool2d(2)

        self.up3  = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(f * 8, f * 4, kernel_size=1),
        )
        self.dec3 = DoubleConvBlock(f * 8, f * 4)

        self.up2  = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(f * 4, f * 2, kernel_size=1),
        )
        self.dec2 = DoubleConvBlock(f * 4, f * 2)

        self.up1  = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(f * 2, f, kernel_size=1),
        )
        self.dec1 = DoubleConvBlock(f * 2, f)

        self.head = nn.Conv2d(f, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.head(d1)


# ============================================================
# 3. Noise Estimation & Re-Corruption
# ============================================================

def estimate_noise_std(image: np.ndarray) -> float:
    """
    Estimate noise std via the Laplacian MAD estimator (Immerkær 1996).

    Applies a discrete Laplacian to the image (zero-mean for flat regions),
    takes the RMS of the interior, then divides by sqrt(20) — the analytical
    normalization for i.i.d. Gaussian noise through a 5-coefficient Laplacian.
    Returns a float in the same units as the (normalized [0,1]) image.
    """
    # Interior slice avoids boundary wrap-around artefacts from np.roll
    lap = (
        np.roll(image,  1, axis=0) + np.roll(image, -1, axis=0) +
        np.roll(image,  1, axis=1) + np.roll(image, -1, axis=1) -
        4.0 * image
    )
    rms  = float(np.sqrt(np.mean(lap[1:-1, 1:-1] ** 2)))
    sigma = rms / np.sqrt(20.0)
    # Clamp to a safe minimum so recorruption noise is never degenerate
    return max(sigma, 1e-4)


def recorrupt_r2r_pair(
    patch: np.ndarray,
    sigma_r: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    R2R asymmetric training pair (Pang et al. CVPR 2021, Theorem 1).

    Samples a single eps ~ N(0, sigma_r² I) and constructs:
        y1 = y + eps,   y2 = y − eps   (alpha=1 special case)

    Because y1 and y2 share the same eps, y2 = 2y − y1 is deterministic
    given y1 and y.  This gives a lower-variance training target than two
    independent draws, tightening the gradient SNR.

    Loss equivalence: E[||f(y1) − y2||²] = E[||f(y1) − x||²] + 4·sigma_r²
    where x is the clean signal — minimising over f yields the same optimum.
    """
    eps = rng.standard_normal(patch.shape).astype(np.float32) * sigma_r
    y1 = np.clip(patch + eps, 0.0, 1.0)
    y2 = np.clip(patch - eps, 0.0, 1.0)
    return y1, y2


def recorrupt_poisson_binomial(
    patch: np.ndarray,
    photon_scale: float,
    binomial_alpha: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    GR2R Binomial splitting for Poisson noise (Monroy et al. CVPR 2025, Sec. 4.2).

    Given the photon count z = round(y · photon_scale):
        omega ~ Binomial(z, binomial_alpha)
        y1 = (y − omega / photon_scale) / (1 − binomial_alpha)
        y2 = y / binomial_alpha − (1 − binomial_alpha) / binomial_alpha · y1

    This preserves the NEF (Natural Exponential Family) structure of Poisson noise
    and is provably unbiased: E[y2 | y] = y.  For low electron-count SEM images
    (z < 20), this substantially reduces training-target variance compared with
    independent Poisson resampling.  Recommended binomial_alpha = 0.15.
    """
    z = np.round(np.maximum(patch * photon_scale, 0.0)).astype(np.int64)
    omega = rng.binomial(z, binomial_alpha).astype(np.float32)
    y1 = np.clip((patch - omega / photon_scale) / (1.0 - binomial_alpha), 0.0, 1.0)
    y2 = np.clip(
        patch / binomial_alpha - (1.0 - binomial_alpha) / binomial_alpha * y1,
        0.0, 1.0,
    )
    return y1, y2


# ============================================================
# 4. GR2R Dataset
# ============================================================

class GR2RDataset(Dataset):
    """
    Self-supervised R2R / GR2R dataset: each item is a (y1, y2) pair.

    Gaussian mode — R2R shared-epsilon pair (Pang et al. 2021):
        eps ~ N(0, sigma_r² I);  y1 = y + eps,  y2 = y − eps
        y2 is deterministic given y1 (lower variance than independent draws).

    Poisson mode — GR2R Binomial splitting (Monroy et al. 2025):
        omega ~ Binomial(round(y·photon_scale), binomial_alpha)
        y1 and y2 derived from the universal GR2R formula; preserves NEF
        structure of SEM shot noise.

    MSE over ALL pixels — no blind-spot masking needed.
    """

    def __init__(
        self,
        image: np.ndarray,
        patch_size:      int   = 64,
        num_patches:     int   = 2000,
        sigma_r:         float = 0.05,   # re-corruption noise std (Gaussian mode)
        photon_scale:    float = 100.0,  # photon scale (Poisson mode)
        binomial_alpha:  float = 0.15,   # Binomial success probability (Poisson mode)
        use_poisson:     bool  = False,
        rng_seed:        int   = None,
    ):
        assert patch_size % 8 == 0, f"patch_size must be divisible by 8, got {patch_size}"
        assert image.shape[0] >= patch_size and image.shape[1] >= patch_size, (
            f"Image shape {image.shape} is smaller than patch_size={patch_size}. "
            "Reduce --patch_size."
        )

        self.image           = image
        self.patch_size      = patch_size
        self.num_patches     = num_patches
        self.sigma_r         = sigma_r
        self.photon_scale    = photon_scale
        self.binomial_alpha  = binomial_alpha
        self.use_poisson     = use_poisson
        self.rng             = np.random.default_rng(rng_seed)
        self.H, self.W       = image.shape

    def __len__(self) -> int:
        return self.num_patches

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        P  = self.patch_size
        r0 = int(self.rng.integers(0, self.H - P))
        c0 = int(self.rng.integers(0, self.W - P))
        patch = self.image[r0:r0 + P, c0:c0 + P].copy()

        if self.use_poisson:
            y1, y2 = recorrupt_poisson_binomial(
                patch, self.photon_scale, self.binomial_alpha, self.rng)
        else:
            y1, y2 = recorrupt_r2r_pair(patch, self.sigma_r, self.rng)

        return (
            torch.from_numpy(y1).unsqueeze(0),   # (1, P, P)  — network input
            torch.from_numpy(y2).unsqueeze(0),   # (1, P, P)  — training target
        )


# ============================================================
# 5. Training Loop
# ============================================================

def train_gr2r(
    model: nn.Module,
    image: np.ndarray,
    patch_size:      int   = 64,
    batch_size:      int   = 128,
    num_epochs:      int   = 100,
    learning_rate:   float = 4e-4,
    sigma_r:         float = 0.05,
    photon_scale:    float = 100.0,
    binomial_alpha:  float = 0.15,
    use_poisson:     bool  = False,
    patches_per_epoch: int = 2000,
    val_fraction:    float = 0.1,
    device: torch.device   = None,
) -> nn.Module:
    """Train R2R/GR2R on a single image."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    n_val   = max(1, int(patches_per_epoch * val_fraction))
    n_train = patches_per_epoch - n_val

    common_kw = dict(
        patch_size=patch_size,
        sigma_r=sigma_r,
        photon_scale=photon_scale,
        binomial_alpha=binomial_alpha,
        use_poisson=use_poisson,
    )
    train_ds = GR2RDataset(image, num_patches=n_train, rng_seed=42,  **common_kw)
    val_ds   = GR2RDataset(image, num_patches=n_val,   rng_seed=99,  **common_kw)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    loss_fn   = nn.MSELoss()   # mean over all pixels (no mask needed in GR2R)

    mode_str = f"Poisson(scale={photon_scale:.0f})" if use_poisson \
               else f"Gaussian(σ_r={sigma_r:.4f})"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nDevice: {device}  |  Model parameters: {n_params:,}")
    print(f"Re-corruption mode: {mode_str}")
    print(f"patch_size={patch_size}  batch_size={batch_size}  epochs={num_epochs}")
    print(f"Patches/epoch: train={n_train}  val={n_val}\n")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        model.train()
        tr_loss, tr_steps = 0.0, 0
        for y1, y2 in train_loader:
            y1 = y1.to(device)
            y2 = y2.to(device)

            optimizer.zero_grad()
            pred = model(y1)
            loss = loss_fn(pred, y2)
            loss.backward()
            optimizer.step()

            tr_loss  += loss.item()
            tr_steps += 1

        model.eval()
        vl_loss, vl_steps = 0.0, 0
        with torch.no_grad():
            for y1, y2 in val_loader:
                y1 = y1.to(device)
                y2 = y2.to(device)
                vl_loss  += loss_fn(model(y1), y2).item()
                vl_steps += 1

        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch:3d}/{num_epochs}]  "
                  f"train={tr_loss / max(tr_steps, 1):.6f}  "
                  f"val={vl_loss / max(vl_steps, 1):.6f}  "
                  f"time={time.time() - t0:.1f}s")

    print("\nTraining complete.")
    return model


# ============================================================
# 6. Tiled Inference — Opt 2 (Batched) + Opt 4 (Reflection Padding)
# ============================================================

def _compute_padding(image_size: int, tile_size: int) -> int:
    """Return reflection padding needed on one axis (per-axis, independent)."""
    pad = max(0, tile_size - image_size)
    padded = image_size + pad
    remainder = padded % 8
    if remainder != 0:
        pad += 8 - remainder
    return pad


def predict_tiled(
    model: nn.Module,
    image: np.ndarray,
    tile_size:        Tuple[int, int] = (256, 256),
    tile_overlap:     Tuple[int, int] = (48, 48),
    infer_batch_size: int             = 8,
    device: torch.device              = None,
) -> np.ndarray:
    """
    Tiled inference with Hann-window blending to avoid seams.

    Opt 4 — Reflection Padding:
        Images smaller than tile_size are padded per-axis independently,
        fixing the if-condition bug where one axis misses divisible-by-8 alignment.

    Opt 2 — Batched GPU Inference:
        Tiles are stacked into batches of `infer_batch_size` for one forward pass,
        maximising CUDA occupancy.

    Returns denoised image as float32 (H, W) — same shape as input.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    H, W   = image.shape
    th, tw = tile_size
    oh, ow = tile_overlap

    assert th % 8 == 0 and tw % 8 == 0, \
        f"tile_size dimensions must be divisible by 8, got {tile_size}"
    assert oh < th and ow < tw, \
        f"tile_overlap {tile_overlap} must be smaller than tile_size {tile_size}"

    # Opt 4: Reflection padding (per-axis, independent)
    pad_h = _compute_padding(H, th)
    pad_w = _compute_padding(W, tw)

    if pad_h > 0 or pad_w > 0:
        padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
    else:
        padded = image

    pH, pW   = padded.shape
    stride_h = th - oh
    stride_w = tw - ow

    hann_h  = torch.hann_window(th, periodic=False).numpy()
    hann_w  = torch.hann_window(tw, periodic=False).numpy()
    hann_2d = np.outer(hann_h, hann_w)

    output_sum = np.zeros((pH, pW), dtype=np.float64)
    weight_sum = np.zeros((pH, pW), dtype=np.float64)

    row_starts: List[int] = list(range(0, pH - th + 1, stride_h))
    col_starts: List[int] = list(range(0, pW - tw + 1, stride_w))
    if row_starts[-1] + th < pH:
        row_starts.append(pH - th)
    if col_starts[-1] + tw < pW:
        col_starts.append(pW - tw)

    coords      = [(r0, c0) for r0 in row_starts for c0 in col_starts]
    total_tiles = len(coords)
    processed   = 0

    # Opt 2: Batched GPU inference
    model.eval()
    with torch.no_grad():
        for i in range(0, total_tiles, infer_batch_size):
            batch_coords = coords[i:i + infer_batch_size]
            tiles   = [padded[r0:r0 + th, c0:c0 + tw] for (r0, c0) in batch_coords]
            batch_t = torch.from_numpy(np.stack(tiles)).unsqueeze(1).to(device)
            preds   = model(batch_t).squeeze(1).cpu().numpy()

            for j, (r0, c0) in enumerate(batch_coords):
                output_sum[r0:r0 + th, c0:c0 + tw] += preds[j].astype(np.float64) * hann_2d
                weight_sum[r0:r0 + th, c0:c0 + tw] += hann_2d

            processed = min(i + infer_batch_size, total_tiles)
            if processed % max(infer_batch_size, total_tiles // 5 or 1) == 0 \
                    or processed == total_tiles:
                print(f"    tiles: {processed}/{total_tiles}")

    denoised_pad = (output_sum / np.maximum(weight_sum, 1e-8)).astype(np.float32)
    return denoised_pad[:H, :W]


# ============================================================
# 7. Save Outputs
# ============================================================

def save_outputs(
    image:    np.ndarray,
    denoised: np.ndarray,
    img_min:  float,
    img_max:  float,
    tif_path: str,
    png_path: str,
) -> None:
    """Save denoised TIF (original value range) and side-by-side comparison PNG."""
    denoised_original = (denoised * (img_max - img_min) + img_min).astype(np.float32)
    tifffile.imwrite(tif_path, denoised_original)
    print(f"  Saved TIF: {tif_path}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image,    cmap='gray'); axes[0].set_title('Original');         axes[0].axis('off')
    axes[1].imshow(denoised, cmap='gray'); axes[1].set_title('GR2R Denoised');    axes[1].axis('off')
    diff = np.abs(image - denoised) * 3
    axes[2].imshow(diff,     cmap='hot');  axes[2].set_title('Difference (×3)'); axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  Saved PNG: {png_path}")


# ============================================================
# 8. Main Pipeline
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="R2R/GR2R SEM denoiser: recorrupted-to-recorrupted, no blind-spot masking."
    )
    parser.add_argument('--input',           type=str,   default='data/test_sem.tif',
                        help='Path to input .tif/.tiff/.png image')
    parser.add_argument('--output',          type=str,   default='',
                        help='Path to output .tif (default: data/denoised_sem_GR2R.tif)')
    parser.add_argument('--epochs',          type=int,   default=100)
    parser.add_argument('--patch_size',      type=int,   default=64)
    parser.add_argument('--batch_size',      type=int,   default=128)
    parser.add_argument('--tile_size',       type=int,   default=256,
                        help='Inference tile size applied to both H and W')
    parser.add_argument('--tile_overlap',    type=int,   default=48)
    parser.add_argument('--alpha',           type=float, default=1.0,
                        help='Gaussian re-corruption strength: σ_r = alpha × σ_est. '
                             'Lower (0.5) is gentler; higher (2.0) adds stronger supervision. '
                             'Not used in --poisson mode.')
    parser.add_argument('--noise_std',       type=float, default=0.0,
                        help='Manual noise std override (0 = auto-estimate from image). '
                             'Set this if you know the noise level precisely.')
    parser.add_argument('--poisson',         action='store_true',
                        help='Use GR2R Binomial-splitting re-corruption instead of Gaussian. '
                             'Recommended when SEM noise is dominated by shot noise.')
    parser.add_argument('--photon_scale',    type=float, default=100.0,
                        help='Photon scale for Poisson re-corruption (used only with --poisson). '
                             'Higher values → finer photon-count resolution. '
                             'Tune to match estimated photon count range of your SEM.')
    parser.add_argument('--binomial_alpha',  type=float, default=0.15,
                        help='Binomial success probability for GR2R Poisson splitting '
                             '(used only with --poisson). Monroy et al. recommend 0.15. '
                             'Smaller values reduce re-corruption variance.')
    parser.add_argument('--mc_samples',      type=int,   default=1,
                        help='Monte Carlo inference samples (GR2R §5.3). '
                             'J > 1 re-corrupts the image J times and averages predictions, '
                             'reducing output variance. Recommended J=5–15 for best quality; '
                             'runtime scales linearly. Default 1 (no MC).')
    parser.add_argument('--device',          type=str,   default=None,
                        help='Device override: cuda, cpu, cuda:1 … (default: auto)')
    args = parser.parse_args()

    os.makedirs('data', exist_ok=True)

    tif_out = args.output if args.output else 'data/denoised_sem_GR2R.tif'
    png_out = (tif_out.replace('.tif', '_comparison.png').replace('.tiff', '_comparison.png')
               if args.output else 'data/denoising_result_GR2R.png')

    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))

    # ── 1. Load image ────────────────────────────────────────────────────────
    print(f"\nLoading: {args.input}")
    image, img_min, img_max = load_sem_image(args.input)
    print(f"  Shape: {image.shape}  |  Range: [{img_min:.2f}, {img_max:.2f}]")

    # ── 2. Estimate / set noise std ──────────────────────────────────────────
    if args.noise_std > 0:
        sigma_est = args.noise_std
        print(f"  Noise std (manual):    σ = {sigma_est:.5f}")
    else:
        sigma_est = estimate_noise_std(image)
        print(f"  Noise std (estimated): σ = {sigma_est:.5f}")

    sigma_r = args.alpha * sigma_est
    print(f"  Re-corruption std:     σ_r = α × σ = {args.alpha:.2f} × {sigma_est:.5f} = {sigma_r:.5f}")

    if args.poisson:
        print(f"  Re-corruption mode: Poisson Binomial "
              f"(photon_scale={args.photon_scale:.0f}, α_b={args.binomial_alpha:.2f})")
    else:
        print(f"  Re-corruption mode: Gaussian (R2R shared-ε pair)")

    # ── 3. Build model ────────────────────────────────────────────────────────
    model = N2VUNet(in_channels=1, base_features=32)

    # ── 4. Train ──────────────────────────────────────────────────────────────
    model = train_gr2r(
        model, image,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        sigma_r=sigma_r,
        photon_scale=args.photon_scale,
        binomial_alpha=args.binomial_alpha,
        use_poisson=args.poisson,
        device=device,
    )

    # ── 5. Inference (optionally Monte Carlo) ─────────────────────────────────
    tile_kw = dict(
        tile_size=(args.tile_size, args.tile_size),
        tile_overlap=(args.tile_overlap, args.tile_overlap),
        device=device,
    )
    if args.mc_samples > 1:
        print(f"\nRunning Monte Carlo tiled inference (J={args.mc_samples})...")
        mc_rng       = np.random.default_rng(0)
        denoised_acc = np.zeros_like(image, dtype=np.float64)
        for j in range(1, args.mc_samples + 1):
            if args.poisson:
                y1_mc, _ = recorrupt_poisson_binomial(
                    image, args.photon_scale, args.binomial_alpha, mc_rng)
            else:
                y1_mc, _ = recorrupt_r2r_pair(image, sigma_r, mc_rng)
            print(f"  MC sample {j}/{args.mc_samples}")
            denoised_acc += predict_tiled(model, y1_mc, **tile_kw).astype(np.float64)
        denoised = (denoised_acc / args.mc_samples).astype(np.float32)
    else:
        print("\nRunning tiled inference...")
        denoised = predict_tiled(model, image, **tile_kw)

    # ── 6. Save ───────────────────────────────────────────────────────────────
    print("\nSaving results...")
    save_outputs(image, denoised, img_min, img_max, tif_out, png_out)

    print(f"\nDone.")


if __name__ == '__main__':
    main()

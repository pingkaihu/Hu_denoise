# ============================================================
# SEM Image Denoising — Noise2Score (pure PyTorch)
# ============================================================
# Based on:
#   "Noise2Score: Tweedie's Approach to Self-Supervised Image Denoising
#    without Clean Images"
#   Kwanyoung Kim, Jong Chul Ye
#   NeurIPS 2021  (arXiv:2106.07009)
#   GitHub: github.com/cubeyoung/Noise2Score
#
# Key idea:
#   An Amortized Residual Denoising Autoencoder (AR-DAE) is trained to
#   estimate the score function ∇_y log p(y) from the noisy image alone.
#   Tweedie's formula then maps the score to a clean estimate:
#
#     Gaussian (σ):  x̂ = y + σ² · R_Θ(y)
#     Poisson  (ζ):  x̂ = (y + ζ/2) · exp(R_Θ(y/ζ))
#     Gamma  (α,β):  x̂ = β·y / ((α−1) − y·R_Θ(y))
#
#   AR-DAE training loss (noise-model agnostic):
#       L = E[ ‖u + σ_a · R_Θ(y + σ_a·u)‖² ],  u ~ N(0, I)
#   At convergence: R_Θ(y) ≈ ∇_y log p(y).
#
# Blind mode (--blind):
#   Pre-computes the score map once, then grid-searches σ over
#   [σ_est/4, σ_est·4] using total-variation as quality proxy.
#   Cost: O(grid_size) Tweedie evaluations — essentially free.
#
# Identical to denoise_GR2R.py:
#   = load_sem_image()       — ITU-R RGB→gray, float32 [0, 1]
#   = DoubleConvBlock, N2VUNet — 4-level encoder-decoder, base_features=32
#   = _compute_padding(), predict_tiled() — reflection padding,
#     Hann-window blending, batched GPU inference (infer_batch_size=8)
#   = save_outputs()         — TIF + 3-panel PNG
#   = Adam + CosineAnnealingLR, lr=4e-4 → 1e-6
#
# Requirements: torch>=2.0.0  tifffile  matplotlib  numpy
# Usage:
#   python denoise_N2Score.py
#   python denoise_N2Score.py --input data/test_sem.tif --epochs 200
#   python denoise_N2Score.py --noise_model poisson --poisson_zeta 0.05
#   python denoise_N2Score.py --blind
#   python denoise_N2Score.py --noise_model gamma --gamma_alpha 10 --gamma_beta 10
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
    """4-level encoder-decoder UNet. In N2Score it outputs the score (residual), not a denoised image."""

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
# 3. Noise Estimation
# ============================================================

def estimate_noise_std(image: np.ndarray) -> float:
    """
    Estimate noise std via the Laplacian MAD estimator (Immerkær 1996).

    Applies a discrete Laplacian to the image (zero-mean for flat regions),
    takes the RMS of the interior, then divides by sqrt(20) — the analytical
    normalization for i.i.d. Gaussian noise through a 5-coefficient Laplacian.
    Returns a float in the same units as the (normalized [0,1]) image.
    """
    lap = (
        np.roll(image,  1, axis=0) + np.roll(image, -1, axis=0) +
        np.roll(image,  1, axis=1) + np.roll(image, -1, axis=1) -
        4.0 * image
    )
    rms   = float(np.sqrt(np.mean(lap[1:-1, 1:-1] ** 2)))
    sigma = rms / np.sqrt(20.0)
    return max(sigma, 1e-4)


# ============================================================
# 4. AR-DAE Dataset
# ============================================================

class N2ScoreDataset(Dataset):
    """
    AR-DAE dataset for Noise2Score (Kim & Ye, NeurIPS 2021).

    Each item augments a random patch with Gaussian noise u ~ N(0, I):
        y_aug = y_patch + σ_a · u

    Training loss:  E[ ‖u + σ_a · R_Θ(y_aug)‖² ]
    At convergence: R_Θ(y) ≈ ∇_y log p(y)  (score function)

    For Poisson mode the network is trained on the ζ-scaled image y/ζ
    so that it learns the score in the natural Poisson parameterization.
    """

    def __init__(
        self,
        image:       np.ndarray,
        patch_size:  int   = 64,
        num_patches: int   = 2000,
        sigma_a:     float = 0.1,
        rng_seed:    int   = None,
    ):
        assert patch_size % 8 == 0, f"patch_size must be divisible by 8, got {patch_size}"
        assert image.shape[0] >= patch_size and image.shape[1] >= patch_size, (
            f"Image shape {image.shape} is smaller than patch_size={patch_size}. "
            "Reduce --patch_size."
        )

        self.image       = image
        self.patch_size  = patch_size
        self.num_patches = num_patches
        self.sigma_a     = sigma_a
        self.rng         = np.random.default_rng(rng_seed)
        self.H, self.W   = image.shape

    def __len__(self) -> int:
        return self.num_patches

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        P  = self.patch_size
        r0 = int(self.rng.integers(0, self.H - P))
        c0 = int(self.rng.integers(0, self.W - P))
        y_patch = self.image[r0:r0 + P, c0:c0 + P].copy()

        u     = self.rng.standard_normal((P, P)).astype(np.float32)
        y_aug = (y_patch + self.sigma_a * u).astype(np.float32)

        return (
            torch.from_numpy(y_aug).unsqueeze(0),  # (1, P, P) — network input
            torch.from_numpy(u).unsqueeze(0),       # (1, P, P) — for AR-DAE loss
        )


# ============================================================
# 5. Training Loop
# ============================================================

def train_n2score(
    model:           nn.Module,
    image:           np.ndarray,
    patch_size:      int   = 64,
    batch_size:      int   = 128,
    num_epochs:      int   = 200,
    learning_rate:   float = 4e-4,
    sigma_a:         float = 0.1,
    patches_per_epoch: int = 2000,
    val_fraction:    float = 0.1,
    device: torch.device   = None,
) -> nn.Module:
    """
    Train the AR-DAE score network on a single noisy image.

    Loss: mean( (u + σ_a · R_Θ(y + σ_a·u))² )
    This is the AR-DAE objective from Kim & Ye 2021, Eq. (9).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    n_val   = max(1, int(patches_per_epoch * val_fraction))
    n_train = patches_per_epoch - n_val

    train_ds = N2ScoreDataset(image, patch_size=patch_size, num_patches=n_train,
                              sigma_a=sigma_a, rng_seed=42)
    val_ds   = N2ScoreDataset(image, patch_size=patch_size, num_patches=n_val,
                              sigma_a=sigma_a, rng_seed=99)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nDevice: {device}  |  Model parameters: {n_params:,}")
    print(f"AR-DAE augmentation σ_a = {sigma_a:.5f}")
    print(f"patch_size={patch_size}  batch_size={batch_size}  epochs={num_epochs}")
    print(f"Patches/epoch: train={n_train}  val={n_val}\n")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        model.train()
        tr_loss, tr_steps = 0.0, 0
        for y_aug, u in train_loader:
            y_aug = y_aug.to(device)
            u     = u.to(device)

            optimizer.zero_grad()
            pred = model(y_aug)              # R_Θ(y + σ_a·u)
            # AR-DAE loss: ‖u + σ_a · R_Θ(y_aug)‖²
            loss = ((u + sigma_a * pred) ** 2).mean()
            loss.backward()
            optimizer.step()

            tr_loss  += loss.item()
            tr_steps += 1

        model.eval()
        vl_loss, vl_steps = 0.0, 0
        with torch.no_grad():
            for y_aug, u in val_loader:
                y_aug = y_aug.to(device)
                u     = u.to(device)
                pred  = model(y_aug)
                vl_loss  += ((u + sigma_a * pred) ** 2).mean().item()
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
# 6. Tiled Inference — returns raw score map (float32, unbounded)
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
    Tiled inference with Hann-window blending.

    For N2Score the model outputs the score R_Θ(y), which is unbounded —
    do NOT clip the output.  apply_tweedie() clips after the formula.

    Opt 4 — Reflection Padding (per-axis independent).
    Opt 2 — Batched GPU Inference (infer_batch_size tiles per forward pass).
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

    score_pad = (output_sum / np.maximum(weight_sum, 1e-8)).astype(np.float32)
    return score_pad[:H, :W]


# ============================================================
# 7. Tweedie's Formula
# ============================================================

def apply_tweedie(
    y:          np.ndarray,
    score_map:  np.ndarray,
    noise_model: str,
    sigma:      float = 0.05,
    sigma_a:    float = 0.0,
    zeta:       float = 0.05,
    alpha:      float = 10.0,
    beta:       float = 10.0,
) -> np.ndarray:
    """
    Apply Tweedie's formula to recover x̂ from score map R_Θ(y).

    The AR-DAE trained at augmentation σ_a learns the score of the distribution
    smoothed by σ_a: s_θ(y) ≈ (E[x|y] - y) / (σ² + σ_a²).
    The correct Tweedie coefficient is therefore (σ² + σ_a²), not just σ²:
        x̂ = y + (σ² + σ_a²) · s_θ(y) = E[x|y, noise=√(σ²+σ_a²)]
    Using only σ² would give a correction factor of σ²/(σ²+σ_a²) << 1,
    leaving most noise intact.

    Gaussian: x̂ = y + (σ² + σ_a²) · score
    Poisson:  x̂ = (y + ζ/2) · exp(score)
              (score was computed at y/ζ input, so no extra ζ division here)
    Gamma:    x̂ = β·y / ((α−1) − y·score)
              Denominator guard prevents division near zero.
    """
    if noise_model == 'gaussian':
        coeff = sigma ** 2 + sigma_a ** 2
        x_hat = y + coeff * score_map

    elif noise_model == 'poisson':
        # score_map was computed on y/ζ; Tweedie Poisson formula (Kim & Ye 2021, Eq. 6)
        x_hat = (y + zeta / 2.0) * np.exp(score_map)

    elif noise_model == 'gamma':
        # Tweedie Gamma formula (Kim & Ye 2021, Eq. 7)
        denom = (alpha - 1.0) - y * score_map
        # Guard: prevent division by values near zero or negative
        denom = np.where(np.abs(denom) < 1e-6, np.sign(denom + 1e-12) * 1e-6, denom)
        x_hat = beta * y / denom

    else:
        raise ValueError(f"Unknown noise_model: {noise_model!r}. Choose gaussian/poisson/gamma.")

    return np.clip(x_hat, 0.0, 1.0).astype(np.float32)


# ============================================================
# 8. Blind σ Search (Gaussian only)
# ============================================================

def tv_norm(img: np.ndarray) -> float:
    """Isotropic total-variation norm (finite differences)."""
    return float(
        np.mean(np.abs(np.diff(img, axis=0))) +
        np.mean(np.abs(np.diff(img, axis=1)))
    )


def blind_sigma_search(
    y:         np.ndarray,
    score_map: np.ndarray,
    sigma_est: float,
    n_grid:    int = 20,
) -> Tuple[float, np.ndarray]:
    """
    Grid-search Tweedie coefficient c = σ² + σ_a² using TV-norm as proxy.

    The score map is fixed after training; evaluating x̂ = y + c·score for
    different c costs only O(n_grid) element-wise operations.  We search c
    around the natural range [σ_est²/4, (σ_est·4)²] and return the value
    minimising total variation of the denoised image.

    Returns (best_c, denoised_image).
    """
    c_lo = max((sigma_est / 4.0) ** 2, 1e-8)
    c_hi = (sigma_est * 4.0) ** 2
    candidates = np.geomspace(c_lo, c_hi, n_grid)

    best_c       = candidates[0]
    best_tv      = float('inf')
    best_denoised = None

    print(f"\nBlind coeff search: {n_grid} candidates  c ∈ [{c_lo:.6f}, {c_hi:.6f}]")
    for c in candidates:
        denoised = np.clip(y + c * score_map, 0.0, 1.0).astype(np.float32)
        tv = tv_norm(denoised)
        if tv < best_tv:
            best_tv      = tv
            best_c       = c
            best_denoised = denoised

    print(f"  Best c = {best_c:.6f}  →  σ_eff ≈ {best_c**0.5:.5f}  (TV = {best_tv:.6f})")
    return best_c, best_denoised


# ============================================================
# 9. Save Outputs
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
    axes[0].imshow(image,    cmap='gray'); axes[0].set_title('Original');              axes[0].axis('off')
    axes[1].imshow(denoised, cmap='gray'); axes[1].set_title('N2Score Denoised');      axes[1].axis('off')
    diff = np.abs(image - denoised) * 3
    axes[2].imshow(diff,     cmap='hot');  axes[2].set_title('Difference (×3)');      axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  Saved PNG: {png_path}")


# ============================================================
# 10. Main Pipeline
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Noise2Score SEM denoiser: AR-DAE score estimation + Tweedie's formula."
    )
    parser.add_argument('--input',        type=str,   default='data/test_sem.tif',
                        help='Path to input .tif/.tiff/.png image')
    parser.add_argument('--output',       type=str,   default='',
                        help='Path to output .tif (default: data/denoised_sem_N2Score.tif)')
    parser.add_argument('--epochs',       type=int,   default=200)
    parser.add_argument('--patch_size',   type=int,   default=64)
    parser.add_argument('--batch_size',   type=int,   default=128)
    parser.add_argument('--base_features', type=int,  default=32)
    parser.add_argument('--tile_size',    type=int,   default=256,
                        help='Inference tile size (applied to both H and W)')
    parser.add_argument('--tile_overlap', type=int,   default=48)

    # AR-DAE augmentation
    parser.add_argument('--sigma_a',      type=float, default=0.0,
                        help='AR-DAE augmentation noise σ_a (0 = 0.5 × estimated σ). '
                             'Must be small relative to actual noise so that the learned score '
                             'approximates the true score.  Effective noise removed = '
                             'sqrt(σ_est² + σ_a²).  Typical range: 0.3–1.0 × estimated noise std.')

    # Noise model selection
    parser.add_argument('--noise_model',  type=str,   default='gaussian',
                        choices=['gaussian', 'poisson', 'gamma'],
                        help='Noise model for Tweedie formula. '
                             'gaussian: additive Gaussian. '
                             'poisson: Poisson shot noise. '
                             'gamma: Gamma (multiplicative) noise.')
    parser.add_argument('--sigma',        type=float, default=0.0,
                        help='Gaussian noise std for Tweedie (0 = auto-estimated). '
                             'Used only with --noise_model gaussian.')
    parser.add_argument('--poisson_zeta', type=float, default=0.05,
                        help='Poisson scale ζ: y ~ Poisson(ζ·x). '
                             'Smaller ζ → more noise. Typical SEM: 0.01–0.1.')
    parser.add_argument('--gamma_alpha',  type=float, default=10.0,
                        help='Gamma shape α (> 1 required for Tweedie). '
                             'Higher α → less noise.')
    parser.add_argument('--gamma_beta',   type=float, default=10.0,
                        help='Gamma rate β.')

    # Blind mode
    parser.add_argument('--blind',        action='store_true',
                        help='Blind Gaussian σ search via TV-norm grid (--noise_model gaussian only). '
                             'Ignores --sigma and finds the best σ automatically.')
    parser.add_argument('--blind_grid',   type=int,   default=20,
                        help='Number of σ candidates in blind search grid.')

    parser.add_argument('--device',       type=str,   default=None,
                        help='Device override: cuda, cpu, cuda:1 … (default: auto)')
    args = parser.parse_args()

    os.makedirs('data', exist_ok=True)

    tif_out = args.output if args.output else 'data/denoised_sem_N2Score.tif'
    png_out = (
        tif_out.replace('.tif', '_comparison.png').replace('.tiff', '_comparison.png')
        if args.output else 'data/denoising_result_N2Score.png'
    )

    device = torch.device(
        args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    )

    # ── 1. Load image ─────────────────────────────────────────────────────────
    print(f"\nLoading: {args.input}")
    image, img_min, img_max = load_sem_image(args.input)
    print(f"  Shape: {image.shape}  |  Range: [{img_min:.2f}, {img_max:.2f}]")

    # ── 2. Estimate noise std ─────────────────────────────────────────────────
    sigma_est = estimate_noise_std(image)
    print(f"  Noise std (estimated): σ_est = {sigma_est:.5f}")

    # AR-DAE augmentation level — keep small so learned score ≈ true score
    sigma_a = args.sigma_a if args.sigma_a > 0.0 else 0.5 * sigma_est
    print(f"  AR-DAE augmentation:   σ_a   = {sigma_a:.5f}")

    # Noise model parameters
    if args.noise_model == 'gaussian':
        sigma_tweedie = args.sigma if args.sigma > 0.0 else sigma_est
        if not args.blind:
            sigma_eff = (sigma_tweedie ** 2 + sigma_a ** 2) ** 0.5
            print(f"  Gaussian Tweedie σ:    {sigma_tweedie:.5f}"
                  + (" (manual)" if args.sigma > 0.0 else " (estimated)")
                  + f"  →  effective removal σ_eff = {sigma_eff:.5f}")
    elif args.noise_model == 'poisson':
        print(f"  Poisson ζ: {args.poisson_zeta:.5f}  (score evaluated at y/ζ)")
    elif args.noise_model == 'gamma':
        print(f"  Gamma α={args.gamma_alpha:.2f}, β={args.gamma_beta:.2f}")
        if args.gamma_alpha <= 1.0:
            raise ValueError("--gamma_alpha must be > 1 for Tweedie's formula to be valid.")

    # ── 3. Prepare input for score network ───────────────────────────────────
    # For Poisson mode the network learns the score at y/ζ
    if args.noise_model == 'poisson':
        image_input = image / args.poisson_zeta
        print(f"  Poisson: training score network on y/ζ "
              f"(range [{image_input.min():.3f}, {image_input.max():.3f}])")
    else:
        image_input = image

    # ── 4. Build and train AR-DAE ─────────────────────────────────────────────
    model = N2VUNet(in_channels=1, base_features=args.base_features)
    model = train_n2score(
        model, image_input,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        sigma_a=sigma_a,
        device=device,
    )

    # ── 5. Tiled inference → score map ───────────────────────────────────────
    print("\nRunning tiled inference (score map)...")
    score_map = predict_tiled(
        model, image_input,
        tile_size=(args.tile_size, args.tile_size),
        tile_overlap=(args.tile_overlap, args.tile_overlap),
        device=device,
    )
    print(f"  Score map range: [{score_map.min():.4f}, {score_map.max():.4f}]")

    # ── 6. Apply Tweedie's formula ────────────────────────────────────────────
    print(f"\nApplying Tweedie's formula  (noise_model={args.noise_model})")
    if args.noise_model == 'gaussian' and args.blind:
        _, denoised = blind_sigma_search(image, score_map, sigma_est, n_grid=args.blind_grid)
    else:
        denoised = apply_tweedie(
            image, score_map,
            noise_model=args.noise_model,
            sigma=sigma_tweedie if args.noise_model == 'gaussian' else 0.0,
            sigma_a=sigma_a if args.noise_model == 'gaussian' else 0.0,
            zeta=args.poisson_zeta,
            alpha=args.gamma_alpha,
            beta=args.gamma_beta,
        )

    # ── 7. Save ───────────────────────────────────────────────────────────────
    print("\nSaving results...")
    save_outputs(image, denoised, img_min, img_max, tif_out, png_out)
    print("\nDone.")


if __name__ == '__main__':
    main()

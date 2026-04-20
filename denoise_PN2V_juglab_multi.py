# ============================================================
# SEM Image Denoising — Probabilistic Noise2Void, Multi-Image (juglab-faithful)
# ============================================================
# Based on: Krull et al., "Probabilistic Noise2Void" (2020)
#   Frontiers in Computer Science, doi:10.3389/fcomp.2020.00005
#   GitHub: github.com/juglab/pn2v
#
# Extends denoise_PN2V_juglab.py to train ONE shared model on multiple images
# acquired under similar SEM conditions, then denoise each image with MMSE.
#
# Multi-image advantages:
#   + Histogram built from N × H × W pixel pairs — richer noise statistics,
#     especially in signal regimes rare in any single image.
#   + Shared UNet trains on a larger patch pool — more stable convergence.
#
# Key design:
#   Noise model  │ Non-parametric 2D histogram (256×256 bins), built from ALL
#                │ training images pooled together; row-normalized → p(y|s)
#   UNet output  │ K=800 samples per pixel (K output channels)
#   Training     │ Negative log-evidence: -log(1/K Σ_k p_hist(y|s_k))
#   Inference    │ Per-image MMSE: Σ p(y|s_k)·s_k / Σ p(y|s_k)
#
# When NOT to use:
#   Images acquired under substantially different SEM conditions (beam energy,
#   magnification, dose). A shared histogram averages noise statistics incorrectly
#   across conditions — use denoise_PN2V_juglab.py per image instead.
#
# CLI (mirrors denoise_PN2V_multi.py):
#   python denoise_PN2V_juglab_multi.py --input_dir ./sem_images --output_dir ./denoised
#   python denoise_PN2V_juglab_multi.py --train_dir ./ref_imgs --input_dir ./targets --output_dir ./out
#   python denoise_PN2V_juglab_multi.py --input_dir ./sem_images --save_model juglab.pt
#   python denoise_PN2V_juglab_multi.py --input_dir ./new_imgs   --load_model juglab.pt
#
# Checkpoint format (--save_model / --load_model):
#   {
#       'model':     UNet state_dict,
#       'histogram': numpy array (n_bins, n_bins),
#       'K':         int,
#       'n_bins':    int,
#   }
#
# Memory note (K=800):
#   (B, K, P, P) UNet output per batch: B=32, K=800, P=64 → ~840 MB GPU.
#   Reduce --batch_size if OOM.  Default: 32.
#
# Requirements: torch>=2.0.0  tifffile  matplotlib  numpy
# ============================================================

import math
import argparse
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import tifffile
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

torch.set_float32_matmul_precision('high')


# ============================================================
# 1. Image Loading / Discovery
# ============================================================

def load_sem_image(path: str) -> Tuple[np.ndarray, float, float]:
    """Load SEM image, normalize to float32 [0,1] grayscale.
    Returns (image, original_min, original_max) for value restoration."""
    img = tifffile.imread(path).astype(np.float32)
    if img.ndim == 3 and img.shape[-1] == 3:
        img = img @ np.array([0.2989, 0.5870, 0.1140])
    img_min, img_max = float(img.min()), float(img.max())
    img = (img - img_min) / (img_max - img_min + 1e-8)
    return img, img_min, img_max


def find_images(dirpath: str) -> List[Path]:
    patterns = ['*.tif', '*.tiff', '*.png']
    paths = []
    for p in patterns:
        paths.extend(Path(dirpath).glob(p))
    if not paths:
        raise FileNotFoundError(f"No .tif/.tiff/.png images found in '{dirpath}'")
    return sorted(paths)


# ============================================================
# 2. 4-Neighbor Mean (signal proxy for histogram building)
# ============================================================

def _compute_4neighbor_mean(image: np.ndarray) -> np.ndarray:
    """4-neighbor mean as a low-noise signal proxy for each pixel."""
    img_t = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
    kernel = torch.zeros(1, 1, 3, 3)
    kernel[0, 0, 0, 1] = 0.25
    kernel[0, 0, 2, 1] = 0.25
    kernel[0, 0, 1, 0] = 0.25
    kernel[0, 0, 1, 2] = 0.25
    with torch.no_grad():
        s_proxy = F.conv2d(img_t, kernel, padding=1)
    return s_proxy.squeeze().numpy()


# ============================================================
# 3. Histogram Noise Model (non-parametric, shared across images)
# ============================================================

class HistogramNoiseModel:
    """
    Non-parametric 2D conditional histogram noise model p(y | s).

    H[i, j] = p(y_j | s_i), rows = signal bins, columns = observation bins,
    both over [0, 1] with n_bins each.  Each row is L1-normalised.

    Multi-image histogram building:
        Pixel pairs (s_proxy, y_obs) are pooled from ALL training images before
        the histogram is accumulated and normalised.  A larger pool gives more
        reliable estimates of the conditional distribution at every signal level,
        especially in signal ranges that are rare in any single image.

    Likelihood query (differentiable in s):
        p(y|s) ≈ H[s_lo, y_bin] * (1−frac) + H[s_hi, y_bin] * frac
        where frac = s_float − s_lo  retains gradient through s.
    """

    def __init__(self, n_bins: int = 256):
        self.n_bins    = n_bins
        self.histogram: Optional[np.ndarray] = None
        self._hist_tensor: Optional[torch.Tensor] = None

    def build(self, images: List[np.ndarray]) -> None:
        """Build histogram by pooling pixel pairs from all images."""
        n    = self.n_bins
        hist = np.zeros((n, n), dtype=np.float64)

        for img in images:
            s_proxy = _compute_4neighbor_mean(img)
            y_flat  = img.flatten()
            s_flat  = s_proxy.flatten()
            s_bins  = np.clip((s_flat * (n - 1)).astype(int), 0, n - 1)
            y_bins  = np.clip((y_flat * (n - 1)).astype(int), 0, n - 1)
            np.add.at(hist, (s_bins, y_bins), 1)

        hist    += 1e-30
        row_sums = np.maximum(hist.sum(axis=1, keepdims=True), 1e-20)
        hist    /= row_sums

        self.histogram    = hist.astype(np.float32)
        self._hist_tensor = None

        n_pairs = sum(img.size for img in images)
        print(f"Histogram built: {n}×{n} bins  "
              f"from {n_pairs:,} pixel pairs  across {len(images)} image(s)")

    def _ensure_tensor(self, device: torch.device) -> torch.Tensor:
        if self._hist_tensor is None or self._hist_tensor.device != device:
            assert self.histogram is not None, "Call build() before using the noise model"
            self._hist_tensor = torch.from_numpy(self.histogram).to(device)
        return self._hist_tensor

    def log_likelihood(self, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """log p(y | s), differentiable in s via bilinear interpolation."""
        hist   = self._ensure_tensor(y.device)
        n_bins = self.n_bins

        s_float = s.clamp(0.0, 1.0) * (n_bins - 1)
        s_lo    = s_float.long().clamp(0, n_bins - 2)
        s_hi    = (s_lo + 1).clamp(0, n_bins - 1)
        frac    = s_float - s_lo.float()

        y_bin = (y.clamp(0.0, 1.0) * (n_bins - 1)).long().clamp(0, n_bins - 1)

        p_lo = hist[s_lo, y_bin]
        p_hi = hist[s_hi, y_bin]
        p    = (p_lo * (1.0 - frac) + p_hi * frac).clamp(min=1e-30)

        return torch.log(p)

    @torch.no_grad()
    def plot(self, save_path: str = "data/histogram_noise_model.png") -> None:
        if self.histogram is None:
            return
        n      = self.n_bins
        y_vals = np.linspace(0, 1, n)
        means  = (self.histogram * y_vals[None, :]).sum(axis=1)
        stds   = np.sqrt(np.maximum(
            (self.histogram * y_vals[None, :] ** 2).sum(axis=1) - means ** 2, 0.0,
        ))
        s_vals = np.linspace(0, 1, n)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        im = axes[0].imshow(
            np.log1p(self.histogram.T * 1000), origin='lower', aspect='auto',
            extent=[0, 1, 0, 1], cmap='hot',
        )
        axes[0].set_xlabel("Signal s"); axes[0].set_ylabel("Observation y")
        axes[0].set_title("log(1 + 1000·H[s,y])  (2D histogram)")
        plt.colorbar(im, ax=axes[0])

        axes[1].plot(s_vals, stds,        color='royalblue', label='noise std σ(s)')
        axes[1].plot(s_vals, means - s_vals, color='tomato', label='mean bias μ(s)−s')
        axes[1].axhline(0, color='k', lw=0.5)
        axes[1].set_xlabel("Signal s"); axes[1].set_ylabel("Value")
        axes[1].set_title("Noise std and mean bias vs. signal")
        axes[1].legend(); axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Histogram plot saved: {save_path}")


# ============================================================
# 4. PN2VUNet — K output channels
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


class PN2VUNet(nn.Module):
    """4-level UNet producing K signal-sample predictions per pixel.
    Input must be spatially divisible by 16."""

    def __init__(self, in_channels: int = 1, base_features: int = 32, K: int = 800):
        super().__init__()
        f      = base_features
        self.K = K

        self.enc1 = DoubleConvBlock(in_channels, f)
        self.enc2 = DoubleConvBlock(f,      f * 2)
        self.enc3 = DoubleConvBlock(f * 2,  f * 4)
        self.enc4 = DoubleConvBlock(f * 4,  f * 8)
        self.pool = nn.MaxPool2d(2)

        self.up3  = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                  nn.Conv2d(f * 8, f * 4, kernel_size=1))
        self.dec3 = DoubleConvBlock(f * 8, f * 4)
        self.up2  = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                  nn.Conv2d(f * 4, f * 2, kernel_size=1))
        self.dec2 = DoubleConvBlock(f * 4, f * 2)
        self.up1  = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                  nn.Conv2d(f * 2, f, kernel_size=1))
        self.dec1 = DoubleConvBlock(f * 2, f)
        self.head = nn.Conv2d(f, K, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)   # (B, K, H, W)


# ============================================================
# 5. Multi-Image N2V Dataset
# ============================================================

class MultiImagePN2VDataset(Dataset):
    """
    Draws random blind-spot patches uniformly across all training images.

    Returns (corrupted_patch, original_noisy_patch, mask) tuples.
    Images smaller than patch_size are skipped with a warning.
    patch_size must be divisible by 16 (4 MaxPool2d layers in PN2VUNet).
    """

    def __init__(
        self,
        images:          List[np.ndarray],
        patch_size:      int   = 64,
        num_patches:     int   = 2000,
        mask_ratio:      float = 0.006,
        neighbor_radius: int   = 5,
        rng_seed:        int   = None,
    ):
        assert patch_size % 16 == 0, f"patch_size must be divisible by 16, got {patch_size}"

        self.images = []
        for i, img in enumerate(images):
            if img.shape[0] >= patch_size and img.shape[1] >= patch_size:
                self.images.append(img)
            else:
                print(f"  [WARNING] Image #{i} shape {img.shape} < patch_size={patch_size} "
                      "— skipped for training.")

        if not self.images:
            raise ValueError(
                f"All images are smaller than patch_size={patch_size}. "
                "Reduce --patch_size or provide larger images."
            )

        self.patch_size      = patch_size
        self.num_patches     = num_patches
        self.neighbor_radius = neighbor_radius
        self.n_masked        = max(1, int(patch_size * patch_size * mask_ratio))
        self.rng             = np.random.default_rng(rng_seed)
        self.shapes          = [(img.shape[0], img.shape[1]) for img in self.images]
        self.n_images        = len(self.images)

    def __len__(self) -> int:
        return self.num_patches

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        P       = self.patch_size
        img_idx = int(self.rng.integers(0, self.n_images))
        H, W    = self.shapes[img_idx]
        r0      = int(self.rng.integers(0, H - P + 1))
        c0      = int(self.rng.integers(0, W - P + 1))
        patch   = self.images[img_idx][r0:r0 + P, c0:c0 + P].copy()
        corrupted, mask = self._apply_n2v_masking(patch)
        return (
            torch.from_numpy(corrupted).unsqueeze(0).float(),
            torch.from_numpy(patch).unsqueeze(0).float(),
            torch.from_numpy(mask).unsqueeze(0).float(),
        )

    def _apply_n2v_masking(
        self, patch: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        P, rad    = self.patch_size, self.neighbor_radius
        corrupted = patch.copy()
        mask      = np.zeros((P, P), dtype=np.float32)

        flat_idx   = self.rng.choice(P * P, size=self.n_masked, replace=False)
        rows, cols = np.unravel_index(flat_idx, (P, P))

        dr = self.rng.integers(-rad, rad + 1, size=self.n_masked)
        dc = self.rng.integers(-rad, rad + 1, size=self.n_masked)

        zero_mask = (dr == 0) & (dc == 0)
        if np.any(zero_mask):
            n_fix  = int(np.sum(zero_mask))
            dr_fix = self.rng.integers(-rad, rad + 1, size=n_fix)
            dc_fix = self.rng.integers(-rad, rad + 1, size=n_fix)
            dr_fix[(dr_fix == 0) & (dc_fix == 0)] = 1
            dr[zero_mask] = dr_fix
            dc[zero_mask] = dc_fix

        nr = np.clip(rows + dr, 0, P - 1)
        nc = np.clip(cols + dc, 0, P - 1)
        corrupted[rows, cols] = patch[nr, nc]
        mask[rows, cols]      = 1.0
        return corrupted, mask


# ============================================================
# 6. Training — Negative Log-Evidence (multi-image)
# ============================================================

def train_pn2v_juglab_multi(
    model:         PN2VUNet,
    noise_model:   HistogramNoiseModel,
    images:        List[np.ndarray],
    patch_size:    int   = 64,
    batch_size:    int   = 32,
    num_epochs:    int   = 100,
    learning_rate: float = 3e-4,
    val_frac:      float = 0.1,
    device:        torch.device = None,
) -> PN2VUNet:
    """
    Train PN2VUNet with the negative log-evidence loss on multiple images.

        L = -mean_masked[ log( (1/K) Σ_k p_hist(y | s_k) ) ]
          = -mean_masked[ logsumexp_k( log p_hist(y | s_k) ) − log(K) ]

    Patches are drawn uniformly across all training images each epoch.
    Patches/epoch scales with image count: max(2000, 500 × n_images).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    n_images          = len(images)
    patches_per_epoch = max(2000, 500 * n_images)
    n_val             = max(1, int(patches_per_epoch * val_frac))
    n_train           = patches_per_epoch - n_val

    train_ds = MultiImagePN2VDataset(images, patch_size=patch_size,
                                     num_patches=n_train, rng_seed=42)
    val_ds   = MultiImagePN2VDataset(images, patch_size=patch_size,
                                     num_patches=n_val,   rng_seed=99)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6,
    )

    K        = model.K
    log_K    = math.log(K)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nDevice: {device}  |  UNet: {n_params:,} params  |  K (samples): {K}")
    print(f"Training on {len(train_ds.images)} image(s)")
    print(f"patch_size={patch_size}  batch_size={batch_size}  epochs={num_epochs}")
    print(f"Patches/epoch: train={n_train}  val={n_val}")
    print(f"Peak GPU per batch: "
          f"~{batch_size * K * patch_size * patch_size * 4 / 1e6:.0f} MB  "
          f"(reduce --batch_size if OOM)\n")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        # ── Training ──
        model.train()
        tr_loss, tr_count = 0.0, 0

        for noisy_in, noisy_tgt, mask in train_loader:
            noisy_in  = noisy_in.to(device)
            noisy_tgt = noisy_tgt.to(device)
            mask      = mask.to(device)

            optimizer.zero_grad()
            pred = model(noisy_in)   # (B, K, P, P)

            mask_2d  = mask.bool().squeeze(1)                     # (B, P, P)
            y_obs    = noisy_tgt.squeeze(1)[mask_2d]              # (N_masked,)
            s_samp   = pred.permute(0, 2, 3, 1)[mask_2d]         # (N_masked, K)

            N_masked = y_obs.shape[0]
            y_exp    = y_obs.unsqueeze(1).expand(N_masked, K)     # (N_masked, K)
            log_liks = noise_model.log_likelihood(
                y_exp.reshape(-1), s_samp.reshape(-1),
            ).reshape(N_masked, K)

            log_ev = torch.logsumexp(log_liks, dim=1) - log_K    # (N_masked,)
            loss   = -log_ev.mean()

            loss.backward()
            optimizer.step()

            tr_loss  += loss.item() * N_masked
            tr_count += N_masked

        # ── Validation ──
        model.eval()
        vl_loss, vl_count = 0.0, 0

        with torch.no_grad():
            for noisy_in, noisy_tgt, mask in val_loader:
                noisy_in  = noisy_in.to(device)
                noisy_tgt = noisy_tgt.to(device)
                mask      = mask.to(device)

                pred     = model(noisy_in)
                mask_2d  = mask.bool().squeeze(1)
                y_obs    = noisy_tgt.squeeze(1)[mask_2d]
                s_samp   = pred.permute(0, 2, 3, 1)[mask_2d]

                N_masked = y_obs.shape[0]
                y_exp    = y_obs.unsqueeze(1).expand(N_masked, K)
                log_liks = noise_model.log_likelihood(
                    y_exp.reshape(-1), s_samp.reshape(-1),
                ).reshape(N_masked, K)
                log_ev   = torch.logsumexp(log_liks, dim=1) - log_K
                vl_loss  += (-log_ev.mean()).item() * N_masked
                vl_count += N_masked

        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch:3d}/{num_epochs}]  "
                  f"train_NLE={tr_loss/max(tr_count,1):.4f}  "
                  f"val_NLE={vl_loss/max(vl_count,1):.4f}  "
                  f"elapsed={time.time()-t0:.1f}s")

    print("Training complete.")
    return model


# ============================================================
# 7. Tiled Inference — MMSE Posterior Mean
# ============================================================

def _compute_padding(image_size: int, tile_size: int, divisor: int = 16) -> int:
    pad       = max(0, tile_size - image_size)
    padded    = image_size + pad
    remainder = padded % divisor
    if remainder != 0:
        pad += divisor - remainder
    return pad


@torch.no_grad()
def _mmse_from_samples(
    pred_K:      torch.Tensor,   # (K, H, W)
    y_tile:      torch.Tensor,   # (H, W)
    noise_model: HistogramNoiseModel,
) -> Tuple[np.ndarray, np.ndarray]:
    """MMSE and prior mean from K network samples for one tile."""
    K, H, W = pred_K.shape
    N       = H * W
    s_flat  = pred_K.reshape(K, N)
    y_flat  = y_tile.reshape(N).unsqueeze(0).expand(K, N)

    log_liks  = noise_model.log_likelihood(
        y_flat.reshape(-1), s_flat.reshape(-1),
    ).reshape(K, N)

    weights   = torch.softmax(log_liks, dim=0)       # (K, N)
    mmse_flat = (weights * s_flat).sum(dim=0)          # (N,)
    prior_flat = s_flat.mean(dim=0)                    # (N,)

    return (mmse_flat.cpu().numpy().reshape(H, W).astype(np.float32),
            prior_flat.cpu().numpy().reshape(H, W).astype(np.float32))


def predict_tiled(
    model:            PN2VUNet,
    image:            np.ndarray,
    noise_model:      HistogramNoiseModel,
    tile_size:        Tuple[int, int] = (256, 256),
    tile_overlap:     Tuple[int, int] = (32, 32),
    infer_batch_size: int             = 2,
    device:           torch.device   = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tiled MMSE inference with Hann-window blending.

    Returns:
        mmse_image  : (H, W) — MMSE posterior mean (primary output)
        prior_image : (H, W) — prior mean (unweighted sample average, diagnostic)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    H, W   = image.shape
    th, tw = tile_size
    oh, ow = tile_overlap

    pad_h = _compute_padding(H, th)
    pad_w = _compute_padding(W, tw)
    padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect') \
             if (pad_h > 0 or pad_w > 0) else image
    pH, pW = padded.shape

    hann_2d = np.outer(
        torch.hann_window(th, periodic=False).numpy(),
        torch.hann_window(tw, periodic=False).numpy(),
    )
    mmse_sum   = np.zeros((pH, pW), dtype=np.float64)
    prior_sum  = np.zeros((pH, pW), dtype=np.float64)
    weight_sum = np.zeros((pH, pW), dtype=np.float64)

    stride_h = th - oh
    stride_w = tw - ow
    row_starts: List[int] = list(range(0, pH - th + 1, stride_h))
    col_starts: List[int] = list(range(0, pW - tw + 1, stride_w))
    if row_starts[-1] + th < pH:
        row_starts.append(pH - th)
    if col_starts[-1] + tw < pW:
        col_starts.append(pW - tw)

    coords      = [(r, c) for r in row_starts for c in col_starts]
    total_tiles = len(coords)

    model.eval()

    for i in range(0, total_tiles, infer_batch_size):
        batch_coords = coords[i:i + infer_batch_size]
        tiles  = [padded[r:r + th, c:c + tw] for r, c in batch_coords]
        batch  = torch.from_numpy(np.stack(tiles)).unsqueeze(1).float().to(device)

        preds = model(batch)   # (B, K, th, tw)

        for j, (r, c) in enumerate(batch_coords):
            y_tile = torch.from_numpy(
                padded[r:r + th, c:c + tw].astype(np.float32)
            ).to(device)
            mmse_t, prior_t = _mmse_from_samples(preds[j], y_tile, noise_model)
            mmse_sum[r:r + th, c:c + tw]  += mmse_t.astype(np.float64)  * hann_2d
            prior_sum[r:r + th, c:c + tw] += prior_t.astype(np.float64) * hann_2d
            weight_sum[r:r + th, c:c + tw] += hann_2d

        done = min(i + infer_batch_size, total_tiles)
        if done % max(infer_batch_size, total_tiles // 5 or 1) == 0 \
                or done == total_tiles:
            print(f"    tiles: {done}/{total_tiles}")

    denom    = np.maximum(weight_sum, 1e-8)
    mmse_img  = (mmse_sum  / denom).astype(np.float32)[:H, :W]
    prior_img = (prior_sum / denom).astype(np.float32)[:H, :W]
    return mmse_img, prior_img


# ============================================================
# 8. Save Per-Image Outputs
# ============================================================

def save_outputs(
    image:      np.ndarray,
    mmse:       np.ndarray,
    prior:      np.ndarray,
    img_min:    float,
    img_max:    float,
    tif_path:   str,
    prior_path: str,
    png_path:   str,
) -> None:
    rng        = img_max - img_min
    mmse_orig  = (mmse  * rng + img_min).astype(np.float32)
    prior_orig = (prior * rng + img_min).astype(np.float32)

    tifffile.imwrite(tif_path,   mmse_orig)
    tifffile.imwrite(prior_path, prior_orig)
    print(f"  Saved: {tif_path}")
    print(f"  Saved: {prior_path}")

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(image, cmap='gray');  axes[0].set_title('Original');              axes[0].axis('off')
    axes[1].imshow(prior, cmap='gray');  axes[1].set_title('Prior mean (avg K)');    axes[1].axis('off')
    axes[2].imshow(mmse,  cmap='gray');  axes[2].set_title('MMSE (juglab PN2V)');    axes[2].axis('off')
    axes[3].imshow(np.abs(image - mmse) * 3, cmap='hot')
    axes[3].set_title('|Original − MMSE| × 3');                                      axes[3].axis('off')
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {png_path}")


# ============================================================
# 9. Main Pipeline
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "PN2V juglab-faithful multi-image: shared histogram + K=800 UNet + "
            "per-image MMSE inference."
        )
    )
    parser.add_argument('--input_dir',    type=str, default='.',
                        help='Directory with images to denoise (also used for training '
                             'unless --train_dir is specified)')
    parser.add_argument('--train_dir',    type=str, default='',
                        help='Optional: separate directory used ONLY for training / '
                             'histogram building. All images in --input_dir are denoised.')
    parser.add_argument('--output_dir',   type=str, default='denoised',
                        help='Directory to write denoised results (default: denoised)')
    parser.add_argument('--n_bins',       type=int, default=256,
                        help='Histogram bins per axis (default: 256)')
    parser.add_argument('--K',            type=int, default=800,
                        help='UNet output samples per pixel (default: 800)')
    parser.add_argument('--epochs',       type=int, default=100)
    parser.add_argument('--patch_size',   type=int, default=64,
                        help='Training patch size; must be divisible by 16')
    parser.add_argument('--batch_size',   type=int, default=32,
                        help='Training batch size (reduce if OOM; default: 32)')
    parser.add_argument('--tile_size',    type=int, default=256,
                        help='Inference tile size (both axes, default: 256)')
    parser.add_argument('--tile_overlap', type=int, default=32)
    parser.add_argument('--infer_batch',  type=int, default=2,
                        help='Inference tile batch size (reduce to 1 if OOM; default: 2)')
    parser.add_argument('--save_model',   type=str, default='',
                        help='Save checkpoint (.pt): UNet weights + histogram + K + n_bins')
    parser.add_argument('--load_model',   type=str, default='',
                        help='Load checkpoint — skips training entirely')
    parser.add_argument('--device',       type=str, default=None,
                        help='Device override: cuda, cpu, cuda:1 … (default: auto)')
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device
        else ('cuda' if torch.cuda.is_available() else 'cpu')
    )

    # ── 1. Discover images ──────────────────────────────────────────────────
    infer_paths = find_images(args.input_dir)
    print(f"Images to denoise ({len(infer_paths)}) from '{args.input_dir}':")
    for p in infer_paths:
        print(f"  {p.name}")

    if args.train_dir:
        train_paths = find_images(args.train_dir)
        print(f"\nTraining images ({len(train_paths)}) from '{args.train_dir}':")
        for p in train_paths:
            print(f"  {p.name}")
    else:
        train_paths = infer_paths

    # ── 2. Load training images ─────────────────────────────────────────────
    print("\nLoading training images ...")
    train_images = []
    for p in train_paths:
        img, img_min, img_max = load_sem_image(str(p))
        train_images.append(img)
        print(f"  {p.name}: shape={img.shape}  "
              f"range=[{img_min:.1f}, {img_max:.1f}]")

    # ── 3. Validate patch_size ──────────────────────────────────────────────
    patch_size = args.patch_size
    if patch_size % 16 != 0:
        patch_size = (patch_size // 16 + 1) * 16
        print(f"patch_size rounded up to {patch_size} (must be divisible by 16)")

    # ── 4. Build or load model ──────────────────────────────────────────────
    if args.load_model and os.path.isfile(args.load_model):
        print(f"\nLoading checkpoint: {args.load_model}")
        ckpt        = torch.load(args.load_model, map_location=device)
        K           = ckpt['K']
        n_bins      = ckpt['n_bins']
        model       = PN2VUNet(in_channels=1, base_features=32, K=K).to(device)
        model.load_state_dict(ckpt['model'])
        noise_model = HistogramNoiseModel(n_bins=n_bins)
        noise_model.histogram = ckpt['histogram']
        noise_model._ensure_tensor(device)
        print(f"  K={K}  n_bins={n_bins}  — skipping training")

    else:
        K      = args.K
        n_bins = args.n_bins

        # Build histogram from all training images
        noise_model = HistogramNoiseModel(n_bins=n_bins)
        noise_model.build(train_images)
        noise_model._ensure_tensor(device)

        # Train shared UNet
        model = PN2VUNet(in_channels=1, base_features=32, K=K)
        train_pn2v_juglab_multi(
            model, noise_model, train_images,
            patch_size=patch_size,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            device=device,
        )

        if args.save_model:
            ckpt = {
                'model':     model.state_dict(),
                'histogram': noise_model.histogram,
                'K':         K,
                'n_bins':    n_bins,
            }
            torch.save(ckpt, args.save_model)
            print(f"Checkpoint saved: {args.save_model}")

    # ── 5. Load inference images (if separate from training) ────────────────
    print("\nLoading inference images ...")
    infer_images, infer_meta = [], []
    for p in infer_paths:
        img, img_min, img_max = load_sem_image(str(p))
        infer_images.append(img)
        infer_meta.append((img_min, img_max))
        if args.train_dir:
            print(f"  {p.name}: shape={img.shape}")

    # ── 6. Save histogram diagnostic ────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    noise_model.plot(save_path=str(out_dir / 'histogram_juglab_multi.png'))

    # ── 7. Denoise each image ────────────────────────────────────────────────
    tile_size    = (args.tile_size, args.tile_size)
    tile_overlap = (args.tile_overlap, args.tile_overlap)

    print(f"\nRunning MMSE inference on {len(infer_paths)} image(s) ...")
    for i, (p, img, (img_min, img_max)) in enumerate(
            zip(infer_paths, infer_images, infer_meta)):
        print(f"\n[{i+1}/{len(infer_paths)}] {p.name}")
        mmse_img, prior_img = predict_tiled(
            model, img, noise_model,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            infer_batch_size=args.infer_batch,
            device=device,
        )
        stem      = p.stem
        tif_path  = str(out_dir / f"{stem}_denoised_pn2v_juglab.tif")
        prior_path = str(out_dir / f"{stem}_prior_pn2v_juglab.tif")
        png_path  = str(out_dir / f"{stem}_comparison_pn2v_juglab.png")
        save_outputs(img, mmse_img, prior_img, img_min, img_max,
                     tif_path, prior_path, png_path)

    print(f"\nDone. All results saved to '{out_dir}/'")


if __name__ == '__main__':
    main()

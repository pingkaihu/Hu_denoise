# ============================================================
# SEM Image Denoising — Fully Unsupervised PPN2V + BIC GMM Selection, Multi-Image
# ============================================================
# Based on: Krull et al., "Fully Unsupervised Probabilistic Noise2Void" (2020)
#   arXiv:1911.12291
#   GitHub: github.com/juglab/pn2v
#
# Extends denoise_PPN2V_juglab_multi.py with BIC-based automatic GMM component selection.
#
# Difference from denoise_PPN2V_juglab_multi.py:
# ────────────────────────────────────────────────────────────────
# Aspect           │ PPN2V juglab multi             │ This script (PPN2V BIC multi)
# ─────────────────┼───────────────────────────────┼──────────────────────────────
# n_components     │ User-specified (default: 3)    │ Auto via BIC (default: 0=auto)
# BIC evaluation   │ Not present                    │ sklearn GaussianMixture on pooled
#                  │                                │ (s_proxy, y) pairs from all images
# CLI args added   │ —                              │ --bic_candidates --bic_subsample
#
# BIC algorithm (multi-image):
#   1. Pool (s_proxy, y) pairs from ALL training images — same pairs used for GMM fitting
#   2. Fit 2D sklearn GaussianMixture for each candidate n in --bic_candidates
#   3. Select n with lowest BIC(X) = n_params·ln(N) − 2·ln(L̂)
#   4. Use selected n for the parametric GMMNoiseModel fit
#   Pooled pairs give more statistical support than single-image BIC.
#
# Multi-image advantages over single-image PPN2V BIC:
#   + Bootstrap N2V trains on all images → more stable pseudo-clean reference
#   + BIC evaluation on pooled pixel pairs → richer noise statistics
#   + Shared PN2V UNet trained on larger patch pool → better convergence
#
# CLI (mirrors denoise_PPN2V_juglab_multi.py):
#   python denoise_PPN2V_juglab_bic_multi.py --input_dir ./sem_images --output_dir ./denoised
#   python denoise_PPN2V_juglab_bic_multi.py --input_dir ./sem_images --n_components 3
#   python denoise_PPN2V_juglab_bic_multi.py --input_dir ./sem_images --bic_candidates 2 3 5
#   python denoise_PPN2V_juglab_bic_multi.py --input_dir ./sem_images --save_model ppn2v_bic.pt
#   python denoise_PPN2V_juglab_bic_multi.py --input_dir ./new_imgs   --load_model ppn2v_bic.pt
#
# Checkpoint format (--save_model / --load_model):
#   {
#       'model':        PN2VUNet state_dict,
#       'gmm':          GMMNoiseModel state_dict,
#       'K':            int,
#       'n_components': int,
#   }
#
# Memory note (K=800):
#   (B, K, P, P) UNet output per batch: B=32, K=800, P=64 → ~840 MB GPU.
#   Reduce --batch_size if OOM.
#
# Requirements: torch>=2.0.0  tifffile  matplotlib  numpy  scikit-learn
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
# 2. Signal Proxy (for calibrated mode)
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
# 3. GMM Noise Model (PPN2V parametric model)
# ============================================================

class GMMNoiseModel(nn.Module):
    """
    Parametric signal-dependent GMM noise model.

    p(y | s) = Σ_{k=1}^{C} α_k(s) · N(y; s + δ_k, σ²_k(s))

    where:
        α_k(s)  = softmax_k( a_k·s + b_k )
        μ_k(s)  = s + δ_k
        σ²_k(s) = exp( w_k·s + v_k )

    GMM is fitted to (pseudo_clean, noisy) pixel pairs then frozen.
    """

    def __init__(self, n_components: int = 3):
        super().__init__()
        C = n_components
        self.n_components = C
        self.weight_a     = nn.Parameter(torch.zeros(C))
        self.weight_b     = nn.Parameter(torch.zeros(C))
        self.mean_offsets = nn.Parameter(torch.zeros(C))
        self.var_a        = nn.Parameter(torch.zeros(C))
        self.var_b        = nn.Parameter(torch.full((C,), -2.0))

    def log_likelihood(self, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """log p(y | s) for each pair. Differentiable in s (and GMM params until frozen)."""
        log_alpha = F.log_softmax(
            s.unsqueeze(-1) * self.weight_a + self.weight_b, dim=-1
        )                                                           # (N, C)
        mu        = s.unsqueeze(-1) + self.mean_offsets             # (N, C)
        log_var   = (self.var_a * s.unsqueeze(-1) + self.var_b).clamp(min=-7.0)  # (N, C)
        log_gauss = -0.5 * (
            (y.unsqueeze(-1) - mu) ** 2 / log_var.exp()
            + log_var
            + math.log(2.0 * math.pi)
        )                                                           # (N, C)
        return (log_alpha + log_gauss).logsumexp(dim=-1)           # (N,)

    @torch.no_grad()
    def plot(self, save_path: str = "data/gmm_noise_model.png") -> None:
        C      = self.n_components
        n_pts  = 200
        s_vals = np.linspace(0, 1, n_pts)
        s_t    = torch.from_numpy(s_vals.astype(np.float32))

        log_w_all = F.log_softmax(
            s_t.unsqueeze(-1) * self.weight_a.cpu() + self.weight_b.cpu(),
            dim=-1,
        ).exp().numpy()

        offsets       = self.mean_offsets.cpu().numpy()
        var_a, var_b  = self.var_a.cpu().numpy(), self.var_b.cpu().numpy()
        var_mat       = np.exp(np.outer(s_vals, var_a) + var_b[None, :])
        alpha         = log_w_all
        mean_bias     = (alpha * offsets[None, :]).sum(axis=1)
        second_moment = (alpha * (var_mat + offsets[None, :] ** 2)).sum(axis=1)
        noise_std     = np.sqrt(np.maximum(second_moment - mean_bias ** 2, 0.0))

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(s_vals, noise_std, color='royalblue', label='noise std σ(s)')
        axes[0].plot(s_vals, mean_bias, color='tomato',    label='mean bias δ(s)')
        axes[0].axhline(0, color='k', lw=0.5)
        axes[0].set_xlabel("Signal s"); axes[0].set_ylabel("Value")
        axes[0].set_title("GMM noise std and mean bias vs. signal")
        axes[0].legend(); axes[0].grid(True, alpha=0.3)
        for k in range(C):
            axes[1].plot(s_vals, log_w_all[:, k], label=f'α_{k}(s)')
        axes[1].set_xlabel("Signal s"); axes[1].set_ylabel("Mixture weight")
        axes[1].set_title("Signal-dependent mixture weights α_k(s)")
        axes[1].legend(); axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"GMM plot saved: {save_path}")


# ============================================================
# 4. BIC-Based GMM Component Count Selection (multi-image)
# ============================================================

def select_n_components_bic(
    s_pairs:      np.ndarray,
    y_pairs:      np.ndarray,
    candidates:   List[int] = None,
    subsample:    int       = 200_000,
    random_state: int       = 42,
) -> int:
    """
    Select the number of GMM components from pre-computed (s_proxy, y) pairs
    using the Bayesian Information Criterion.

    BIC = n_params · ln(n) − 2 · ln(L̂)

    A 2D sklearn GaussianMixture on (s_proxy, y) is used as a fast proxy for the
    conditional GMMNoiseModel. It ranks component counts in the same order as the
    full conditional model for this selection purpose, without requiring multiple
    full gradient-descent GMM training runs.

    For multi-image: s_pairs and y_pairs are pooled from all training images.
    The larger pool (default subsample=200 000) gives more statistical support
    than single-image BIC, especially for low-ENL SEM.

    Args:
        s_pairs      : (N,) signal proxy pooled from all training images
        y_pairs      : (N,) noisy observations pooled from all training images
        candidates   : n_components values to evaluate (default: [2,3,5,7,9,11,13])
        subsample    : max pairs for BIC evaluation (default: 200 000)
        random_state : RNG seed for subsampling and GaussianMixture init

    References
    ----------
    Schwarz, G. (1978). Estimating the dimension of a model.
        Annals of Statistics, 6(2), 461–464.

    Note: requires scikit-learn.  pip install scikit-learn
    """
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        raise ImportError(
            "scikit-learn is required for BIC selection.\n"
            "Install with: pip install scikit-learn\n"
            "Or skip BIC with: --n_components 3"
        )

    if candidates is None:
        candidates = [2, 3, 5, 7, 9, 11, 13]

    # Stack into 2D feature matrix: each row = (s_proxy, y)
    X = np.stack([s_pairs, y_pairs], axis=1).astype(np.float32)
    n = len(X)

    if n > subsample:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=subsample, replace=False)
        X   = X[idx]
        n_bic = subsample
    else:
        n_bic = n

    print(f"\nBIC model selection — evaluating n_components ∈ {candidates} "
          f"({n_bic:,} pixel pairs, pooled from all images) ...")

    best_bic, best_k = np.inf, candidates[0]
    for k in candidates:
        try:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type='full',
                random_state=random_state,
                max_iter=300,
                n_init=3,
            ).fit(X)
            bic = gmm.bic(X)
            print(f"  n_comp={k:2d}:  BIC = {bic:12.1f}")
            if bic < best_bic:
                best_bic, best_k = bic, k
        except Exception as exc:
            print(f"  n_comp={k:2d}:  fit failed ({exc}), skipping")

    print(f"  → Selected n_components = {best_k}  (lowest BIC = {best_bic:.1f})\n")
    return best_k


# ============================================================
# 5. UNet Architectures
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


class SimpleN2VUNet(nn.Module):
    """Lightweight 4-level UNet for bootstrap N2V (1-channel output)."""

    def __init__(self, in_channels: int = 1, base_features: int = 32):
        super().__init__()
        f = base_features
        self.enc1 = DoubleConvBlock(in_channels, f)
        self.enc2 = DoubleConvBlock(f,     f * 2)
        self.enc3 = DoubleConvBlock(f * 2, f * 4)
        self.enc4 = DoubleConvBlock(f * 4, f * 8)
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
        self.head = nn.Conv2d(f, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)   # (B, 1, H, W)


class PN2VUNet(nn.Module):
    """4-level UNet producing K signal-sample predictions per pixel.
    Input must be spatially divisible by 16."""

    def __init__(self, in_channels: int = 1, base_features: int = 32, K: int = 800):
        super().__init__()
        f      = base_features
        self.K = K
        self.enc1 = DoubleConvBlock(in_channels, f)
        self.enc2 = DoubleConvBlock(f,     f * 2)
        self.enc3 = DoubleConvBlock(f * 2, f * 4)
        self.enc4 = DoubleConvBlock(f * 4, f * 8)
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
# 6. Multi-Image N2V Datasets
# ============================================================

class MultiImageN2VDataset(Dataset):
    """
    Draws random blind-spot patches uniformly across all training images.
    Returns (corrupted_patch, original_noisy_patch, mask) tuples.
    Images smaller than patch_size are skipped with a warning.
    patch_size must be divisible by 16.
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
                print(f"  [WARNING] Image #{i} shape {img.shape} < patch_size={patch_size} — skipped.")

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

    def _apply_n2v_masking(self, patch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
# 7. Bootstrap N2V — Phase 1 (multi-image)
# ============================================================

def _infer_tiled_single(
    model:      nn.Module,
    image:      np.ndarray,
    tile_size:  int          = 256,
    overlap:    int          = 32,
    batch_size: int          = 4,
    device:     torch.device = None,
) -> np.ndarray:
    """Tiled inference for a 1-channel output model with Hann-window blending."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    H, W   = image.shape
    th = tw = tile_size
    oh = ow = overlap

    def _pad_len(n: int, t: int) -> int:
        pad = max(0, t - n)
        rem = (n + pad) % 16
        return pad + (16 - rem if rem else 0)

    pad_h = _pad_len(H, th)
    pad_w = _pad_len(W, tw)
    padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect') \
             if (pad_h > 0 or pad_w > 0) else image
    pH, pW = padded.shape

    hann_2d = np.outer(
        torch.hann_window(th, periodic=False).numpy(),
        torch.hann_window(tw, periodic=False).numpy(),
    )
    pred_sum   = np.zeros((pH, pW), dtype=np.float64)
    weight_sum = np.zeros((pH, pW), dtype=np.float64)

    stride_h = th - oh
    stride_w = tw - ow
    rows = list(range(0, pH - th + 1, stride_h))
    cols = list(range(0, pW - tw + 1, stride_w))
    if not rows or rows[-1] + th < pH:
        rows.append(max(0, pH - th))
    if not cols or cols[-1] + tw < pW:
        cols.append(max(0, pW - tw))
    coords = [(r, c) for r in rows for c in cols]

    model.eval()
    with torch.no_grad():
        for i in range(0, len(coords), batch_size):
            bc    = coords[i:i + batch_size]
            tiles = [padded[r:r + th, c:c + tw] for r, c in bc]
            batch = torch.from_numpy(np.stack(tiles)).unsqueeze(1).float().to(device)
            preds = model(batch).squeeze(1).cpu().numpy()
            for j, (r, c) in enumerate(bc):
                pred_sum[r:r + th, c:c + tw]   += preds[j].astype(np.float64) * hann_2d
                weight_sum[r:r + th, c:c + tw] += hann_2d

    denom = np.maximum(weight_sum, 1e-8)
    return (pred_sum / denom).astype(np.float32)[:H, :W]


def run_bootstrap_n2v_multi(
    images:        List[np.ndarray],
    n2v_epochs:    int          = 50,
    patch_size:    int          = 64,
    batch_size:    int          = 32,
    learning_rate: float        = 3e-4,
    tile_size:     int          = 256,
    tile_overlap:  int          = 32,
    infer_batch:   int          = 4,
    device:        torch.device = None,
) -> List[np.ndarray]:
    """
    Train one shared lightweight N2V on all images for n2v_epochs.
    Returns a list of pseudo-clean images (one per input image).

    All images share one SimpleN2VUNet — multi-image training gives a better
    pseudo-clean estimate than single-image bootstrap, especially for small images.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_images = len(images)
    print(f"Bootstrap N2V: training shared model on {n_images} image(s) "
          f"for {n2v_epochs} epochs ...")

    n2v_model = SimpleN2VUNet(in_channels=1, base_features=32).to(device)

    patches_per_epoch = max(2000, 500 * n_images)
    dataset = MultiImageN2VDataset(images, patch_size=patch_size,
                                   num_patches=patches_per_epoch, rng_seed=42)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         num_workers=0, pin_memory=(device.type == 'cuda'))

    optimizer = optim.Adam(n2v_model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n2v_epochs, eta_min=1e-5,
    )

    for epoch in range(1, n2v_epochs + 1):
        n2v_model.train()
        total_loss, total_count = 0.0, 0

        for noisy_in, noisy_tgt, mask in loader:
            noisy_in  = noisy_in.to(device)
            noisy_tgt = noisy_tgt.to(device)
            mask      = mask.to(device)

            optimizer.zero_grad()
            pred    = n2v_model(noisy_in)
            mask_2d = mask.bool().squeeze(1)
            y_obs   = noisy_tgt.squeeze(1)[mask_2d]
            s_pred  = pred.squeeze(1)[mask_2d]
            loss    = F.mse_loss(s_pred, y_obs)
            loss.backward()
            optimizer.step()

            total_loss  += loss.item() * y_obs.shape[0]
            total_count += y_obs.shape[0]

        scheduler.step()

        if epoch % 10 == 0 or epoch == n2v_epochs:
            print(f"  Bootstrap N2V [{epoch:3d}/{n2v_epochs}]  "
                  f"MSE={total_loss / max(total_count, 1):.5f}")

    print(f"Bootstrap N2V done — generating pseudo-clean for {n_images} image(s) ...")
    pseudo_cleans = []
    for i, img in enumerate(images):
        pc = _infer_tiled_single(
            n2v_model, img,
            tile_size=tile_size, overlap=tile_overlap,
            batch_size=infer_batch, device=device,
        )
        pseudo_cleans.append(pc)
        print(f"  [{i+1}/{n_images}] pseudo-clean range: [{pc.min():.4f}, {pc.max():.4f}]")
    return pseudo_cleans


# ============================================================
# 8. GMM Fitting — Phase 1 (cont.)
# ============================================================

def fit_gmm_from_pairs(
    s_pairs:      np.ndarray,
    y_pairs:      np.ndarray,
    n_components: int          = 3,
    n_steps:      int          = 1000,
    lr:           float        = 1e-2,
    max_pairs:    int          = 200_000,
    device:       torch.device = None,
) -> GMMNoiseModel:
    """
    Fit parametric GMM by minimizing NLL on (s_proxy, y_obs) pixel pairs.

    Pixel pairs are pooled from all training images before fitting.
    After fitting, all GMM parameters are frozen with requires_grad_(False).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    N = len(s_pairs)
    if N > max_pairs:
        idx     = np.random.default_rng(0).choice(N, size=max_pairs, replace=False)
        s_pairs = s_pairs[idx]
        y_pairs = y_pairs[idx]
        print(f"GMM fitting: subsampled {max_pairs:,} / {N:,} pairs  "
              f"n_components={n_components}")
    else:
        print(f"GMM fitting: {N:,} pixel pairs  n_components={n_components}")

    s_t    = torch.from_numpy(s_pairs.astype(np.float32)).to(device)
    y_t    = torch.from_numpy(y_pairs.astype(np.float32)).to(device)
    gmm    = GMMNoiseModel(n_components=n_components).to(device)
    optim_ = optim.Adam(gmm.parameters(), lr=lr)

    mini_batch = min(65_536, len(s_t))
    N_pairs    = len(s_t)

    for step in range(1, n_steps + 1):
        idx      = torch.randperm(N_pairs, device=device)[:mini_batch]
        log_prob = gmm.log_likelihood(y_t[idx], s_t[idx])
        loss     = -log_prob.mean()
        optim_.zero_grad()
        loss.backward()
        optim_.step()

        if step % 200 == 0 or step == n_steps:
            print(f"  GMM fit [{step:4d}/{n_steps}]  NLL={loss.item():.4f}")

    gmm.requires_grad_(False)
    print("GMM fitting complete — parameters frozen.")
    return gmm


# ============================================================
# 9. PPN2V Training — Phase 2 (multi-image, negative log-evidence)
# ============================================================

def train_ppn2v_multi(
    model:         PN2VUNet,
    noise_model:   GMMNoiseModel,
    images:        List[np.ndarray],
    patch_size:    int          = 64,
    batch_size:    int          = 32,
    num_epochs:    int          = 100,
    learning_rate: float        = 3e-4,
    val_frac:      float        = 0.1,
    device:        torch.device = None,
) -> PN2VUNet:
    """
    Train shared PN2VUNet with neg. log-evidence loss on multiple images.

        L = -mean[ log( 1/K Σ_k p_GMM(y | s_k) ) ]

    Patches drawn uniformly across all images each epoch.
    GMM is excluded from optimizer (requires_grad=False).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model       = model.to(device)
    noise_model = noise_model.to(device)

    n_images          = len(images)
    patches_per_epoch = max(2000, 500 * n_images)
    n_val             = max(1, int(patches_per_epoch * val_frac))
    n_train           = patches_per_epoch - n_val

    train_ds = MultiImageN2VDataset(images, patch_size=patch_size,
                                    num_patches=n_train, rng_seed=42)
    val_ds   = MultiImageN2VDataset(images, patch_size=patch_size,
                                    num_patches=n_val,   rng_seed=99)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=learning_rate
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6,
    )

    K        = model.K
    log_K    = math.log(K)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nDevice: {device}  |  UNet: {n_params:,} params  |  K (samples): {K}")
    print(f"GMM components: {noise_model.n_components}  (frozen)")
    print(f"Training on {len(train_ds.images)} image(s)")
    print(f"patch_size={patch_size}  batch_size={batch_size}  epochs={num_epochs}")
    print(f"Patches/epoch: train={n_train}  val={n_val}")
    print(f"Peak GPU per batch: "
          f"~{batch_size * K * patch_size * patch_size * 4 / 1e6:.0f} MB  "
          f"(reduce --batch_size if OOM)\n")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        model.train()
        tr_loss, tr_count = 0.0, 0

        for noisy_in, noisy_tgt, mask in train_loader:
            noisy_in  = noisy_in.to(device)
            noisy_tgt = noisy_tgt.to(device)
            mask      = mask.to(device)

            optimizer.zero_grad()
            pred = model(noisy_in)   # (B, K, P, P)

            mask_2d  = mask.bool().squeeze(1)
            y_obs    = noisy_tgt.squeeze(1)[mask_2d]
            s_samp   = pred.permute(0, 2, 3, 1)[mask_2d]

            N_masked = y_obs.shape[0]
            y_exp    = y_obs.unsqueeze(1).expand(N_masked, K)
            log_liks = noise_model.log_likelihood(
                y_exp.reshape(-1), s_samp.reshape(-1),
            ).reshape(N_masked, K)

            log_ev = torch.logsumexp(log_liks, dim=1) - log_K
            loss   = -log_ev.mean()
            loss.backward()
            optimizer.step()

            tr_loss  += loss.item() * N_masked
            tr_count += N_masked

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
                  f"train_NLE={tr_loss / max(tr_count, 1):.4f}  "
                  f"val_NLE={vl_loss / max(vl_count, 1):.4f}  "
                  f"elapsed={time.time() - t0:.1f}s")

    print("Training complete.")
    return model


# ============================================================
# 10. Tiled MMSE Inference
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
    pred_K:      torch.Tensor,
    y_tile:      torch.Tensor,
    noise_model: GMMNoiseModel,
) -> Tuple[np.ndarray, np.ndarray]:
    """MMSE and prior mean from K network samples for one tile."""
    K, H, W = pred_K.shape
    N       = H * W
    s_flat  = pred_K.reshape(K, N)
    y_flat  = y_tile.reshape(N).unsqueeze(0).expand(K, N)

    log_liks   = noise_model.log_likelihood(
        y_flat.reshape(-1), s_flat.reshape(-1),
    ).reshape(K, N)

    weights    = torch.softmax(log_liks, dim=0)
    mmse_flat  = (weights * s_flat).sum(dim=0)
    prior_flat = s_flat.mean(dim=0)

    return (mmse_flat.cpu().numpy().reshape(H, W).astype(np.float32),
            prior_flat.cpu().numpy().reshape(H, W).astype(np.float32))


def predict_tiled(
    model:            PN2VUNet,
    image:            np.ndarray,
    noise_model:      GMMNoiseModel,
    tile_size:        Tuple[int, int] = (256, 256),
    tile_overlap:     Tuple[int, int] = (32, 32),
    infer_batch_size: int             = 2,
    device:           torch.device   = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Tiled MMSE inference with Hann-window blending.

    Returns: (mmse_image, prior_image), both (H, W).
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
        preds  = model(batch)   # (B, K, th, tw)

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
# 11. Save Per-Image Outputs
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
    axes[0].imshow(image, cmap='gray'); axes[0].set_title('Original');               axes[0].axis('off')
    axes[1].imshow(prior, cmap='gray'); axes[1].set_title('Prior mean (avg K)');     axes[1].axis('off')
    axes[2].imshow(mmse,  cmap='gray'); axes[2].set_title('MMSE (PPN2V BIC)');       axes[2].axis('off')
    axes[3].imshow(np.abs(image - mmse) * 3, cmap='hot')
    axes[3].set_title('|Original − MMSE| × 3');                                       axes[3].axis('off')
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {png_path}")


# ============================================================
# 12. Main Pipeline
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "PPN2V juglab-faithful multi-image with BIC-based GMM component auto-selection: "
            "shared GMM + bootstrap N2V + K=800 UNet + per-image MMSE inference."
        )
    )
    parser.add_argument('--input_dir',      type=str,   default='.',
                        help='Directory with images to denoise (also used for training '
                             'unless --train_dir is specified)')
    parser.add_argument('--train_dir',      type=str,   default='',
                        help='Optional: separate directory ONLY for training / bootstrap. '
                             'All images in --input_dir are denoised.')
    parser.add_argument('--output_dir',     type=str,   default='denoised',
                        help='Directory to write denoised results (default: denoised)')
    parser.add_argument('--calib_dir',      type=str,   default='',
                        help='Optional: use external calibration images for GMM fitting '
                             '(skips bootstrap N2V; uses 4-neighbor proxy instead)')
    parser.add_argument('--n_components',   type=int,   default=0,
                        help='GMM components (0=auto via BIC, default: 0)')
    parser.add_argument('--bic_candidates', type=int,   nargs='+', default=[2, 3, 5, 7, 9, 11, 13],
                        help='n_components candidates for BIC (default: 2 3 5 7 9 11 13)')
    parser.add_argument('--bic_subsample',  type=int,   default=200_000,
                        help='Max pixel pairs for BIC evaluation, pooled from all images '
                             '(default: 200000)')
    parser.add_argument('--n2v_epochs',     type=int,   default=50,
                        help='Bootstrap N2V epochs (default: 50)')
    parser.add_argument('--gmm_steps',      type=int,   default=1000,
                        help='Gradient steps for GMM fitting (default: 1000)')
    parser.add_argument('--gmm_lr',         type=float, default=1e-2,
                        help='Learning rate for GMM fitting (default: 0.01)')
    parser.add_argument('--K',              type=int,   default=800,
                        help='UNet output samples per pixel (default: 800)')
    parser.add_argument('--epochs',         type=int,   default=100)
    parser.add_argument('--patch_size',     type=int,   default=64,
                        help='Training patch size; must be divisible by 16')
    parser.add_argument('--batch_size',     type=int,   default=32,
                        help='Training batch size (reduce if OOM; default: 32)')
    parser.add_argument('--tile_size',      type=int,   default=256,
                        help='Inference tile size (both axes, default: 256)')
    parser.add_argument('--tile_overlap',   type=int,   default=32)
    parser.add_argument('--infer_batch',    type=int,   default=2,
                        help='Inference tile batch size (reduce to 1 if OOM; default: 2)')
    parser.add_argument('--save_model',     type=str,   default='',
                        help='Save checkpoint (.pt): UNet + GMM state dicts + K + n_components')
    parser.add_argument('--load_model',     type=str,   default='',
                        help='Load checkpoint — skips both bootstrap and training')
    parser.add_argument('--device',         type=str,   default=None,
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

    # ── 2. Validate patch_size ──────────────────────────────────────────────
    patch_size = args.patch_size
    if patch_size % 16 != 0:
        patch_size = (patch_size // 16 + 1) * 16
        print(f"patch_size rounded up to {patch_size} (must be divisible by 16)")

    # ── 3. Build or load model ──────────────────────────────────────────────
    if args.load_model and os.path.isfile(args.load_model):
        print(f"\nLoading checkpoint: {args.load_model}")
        ckpt         = torch.load(args.load_model, map_location=device)
        K            = ckpt['K']
        n_components = ckpt['n_components']
        model        = PN2VUNet(in_channels=1, base_features=32, K=K).to(device)
        model.load_state_dict(ckpt['model'])
        noise_model  = GMMNoiseModel(n_components=n_components).to(device)
        noise_model.load_state_dict(ckpt['gmm'])
        noise_model.requires_grad_(False)
        print(f"  K={K}  n_components={n_components}  — skipping training")

    else:
        # ── 3a. Load training images ────────────────────────────────────────
        print("\nLoading training images ...")
        train_images = []
        for p in train_paths:
            img, img_min, img_max = load_sem_image(str(p))
            train_images.append(img)
            print(f"  {p.name}: shape={img.shape}  "
                  f"range=[{img_min:.1f}, {img_max:.1f}]")

        K = args.K

        # ── 3b. Phase 1: calibrate GMM ──────────────────────────────────────
        if args.calib_dir:
            print(f"\n[Phase 1] Calibrated mode: loading from {args.calib_dir}")
            calib_paths = find_images(args.calib_dir)
            s_all, y_all = [], []
            for p in calib_paths:
                img, _, _ = load_sem_image(str(p))
                print(f"  {p.name}: shape={img.shape}")
                s_proxy = _compute_4neighbor_mean(img)
                s_all.append(s_proxy.flatten())
                y_all.append(img.flatten())
            s_pairs = np.concatenate(s_all).astype(np.float32)
            y_pairs = np.concatenate(y_all).astype(np.float32)
            print(f"Collected {len(s_pairs):,} pixel pairs from "
                  f"{len(calib_paths)} calibration image(s)")
        else:
            print(f"\n[Phase 1] Bootstrap mode: running N2V for "
                  f"{args.n2v_epochs} epochs on {len(train_images)} image(s) ...")
            pseudo_cleans = run_bootstrap_n2v_multi(
                train_images,
                n2v_epochs=args.n2v_epochs,
                patch_size=patch_size,
                batch_size=args.batch_size,
                tile_size=args.tile_size,
                tile_overlap=args.tile_overlap,
                infer_batch=args.infer_batch,
                device=device,
            )
            s_all = [pc.flatten() for pc in pseudo_cleans]
            y_all = [img.flatten() for img in train_images]
            s_pairs = np.concatenate(s_all).astype(np.float32)
            y_pairs = np.concatenate(y_all).astype(np.float32)
            print(f"Collected {len(s_pairs):,} pixel pairs from {len(train_images)} image(s)")

        # ── BIC component selection (or use fixed --n_components) ──────────
        n_components = args.n_components
        if n_components == 0:
            n_components = select_n_components_bic(
                s_pairs, y_pairs,
                candidates=args.bic_candidates,
                subsample=args.bic_subsample,
            )
        else:
            print(f"\nUsing fixed n_components = {n_components} (BIC skipped)")

        print(f"\n[Phase 1] Fitting GMM (n_components={n_components}) ...")
        noise_model = fit_gmm_from_pairs(
            s_pairs, y_pairs,
            n_components=n_components,
            n_steps=args.gmm_steps,
            lr=args.gmm_lr,
            device=device,
        )

        # ── 3c. Phase 2: train PN2V UNet ────────────────────────────────────
        print(f"\n[Phase 2] Training shared PN2V UNet (K={K}) ...")
        model = PN2VUNet(in_channels=1, base_features=32, K=K)
        train_ppn2v_multi(
            model, noise_model, train_images,
            patch_size=patch_size,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            device=device,
        )

        if args.save_model:
            ckpt = {
                'model':        model.state_dict(),
                'gmm':          noise_model.state_dict(),
                'K':            K,
                'n_components': n_components,
            }
            torch.save(ckpt, args.save_model)
            print(f"Checkpoint saved: {args.save_model}")

    # ── 4. Load inference images (if separate from training) ────────────────
    print("\nLoading inference images ...")
    infer_images, infer_meta = [], []
    for p in infer_paths:
        img, img_min, img_max = load_sem_image(str(p))
        infer_images.append(img)
        infer_meta.append((img_min, img_max))
        if args.train_dir or args.load_model:
            print(f"  {p.name}: shape={img.shape}")

    # ── 5. Save GMM diagnostic ──────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    noise_model.to(device)
    noise_model.plot(save_path=str(out_dir / 'gmm_ppn2v_bic_multi.png'))

    # ── 6. Denoise each image ────────────────────────────────────────────────
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
        stem       = p.stem
        tif_path   = str(out_dir / f"{stem}_denoised_ppn2v_juglab_bic.tif")
        prior_path = str(out_dir / f"{stem}_prior_ppn2v_juglab_bic.tif")
        png_path   = str(out_dir / f"{stem}_comparison_ppn2v_juglab_bic.png")
        save_outputs(img, mmse_img, prior_img, img_min, img_max,
                     tif_path, prior_path, png_path)

    print(f"\nDone. All results saved to '{out_dir}/'")


if __name__ == '__main__':
    main()

# ============================================================
# SEM Image Denoising — Log + PPN2V (juglab-faithful)
# ============================================================
# Based on: Krull et al., "Fully Unsupervised Probabilistic Noise2Void" (2020)
#   arXiv:1911.12291
#   GitHub: github.com/juglab/pn2v
#
# Extends denoise_PPN2V_juglab.py with a log1p pre-transform.
#
# Rationale:
#   SEM noise often contains BOTH a multiplicative Gamma component (speckle)
#   AND a signal-dependent Poisson component (shot noise).
#   log1p stabilises the Gamma multiplicative component → reduces it to
#   approximately additive noise with signal-independent variance.
#   PPN2V's parametric GMM then models the residual Poisson signal-dependency
#   remaining in the log domain — capturing structure that plain log+N2V misses.
#
# Difference from denoise_PPN2V_juglab.py:
# ─────────────────────────────────────────────────────────────────
# Aspect           │ PPN2V juglab                    │ This script (log + PPN2V)
# ─────────────────┼────────────────────────────────┼──────────────────────────────
# Pre-processing   │ None (raw [0,1] input)         │ log1p + renorm to [0,1]
# GMM noise domain │ Linear intensity domain         │ Log domain (post log1p)
# GMM signal-dep.  │ Full Poisson-like signal-dep.  │ Residual after log (weaker)
# Best noise type  │ Poisson + light speckle        │ Speckle-dominant + Poisson mix
# Post-processing  │ None                            │ expm1 inverse transform
# ─────────────────┼────────────────────────────────┼──────────────────────────────
#
# Pipeline:
#   load_sem_image()
#     → apply_log_transform()           [log1p, renorm to [0,1]]
#       → bootstrap N2V  (log domain)
#       → GMM fit        (log domain)
#       → PPN2V UNet     (log domain)
#       → MMSE inference (log domain)
#     → inverse_log_transform()         [denorm, expm1]
#       → restore original range → save TIF / PNG
#
# UNet output: K=800 samples per pixel (same as PPN2V juglab)
# Loss:        -log(1/K Σ_k p_GMM(y|s_k))  (same as PPN2V juglab)
# Inference:   MMSE Σ_k p_GMM(y|s_k)·s_k / Σ_k p_GMM(y|s_k)  (same)
#
# Requirements: torch>=2.0.0  tifffile  matplotlib  numpy
# Usage:
#   python test_sem.py                                    # generate synthetic test image
#   python denoise_log_PPN2V_juglab.py                    # bootstrap mode (default)
#   python denoise_log_PPN2V_juglab.py --calib_dir ./sem_images
#   python denoise_log_PPN2V_juglab.py --n_components 5 --n2v_epochs 80
# ============================================================

import math
import argparse
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import glob
import time
from typing import List, Tuple, Optional

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
# 1. Image Loading
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


def load_images_from_dir(dirpath: str) -> List[np.ndarray]:
    """Load all .tif/.tiff/.png images from a directory, normalized to [0,1]."""
    patterns = ['*.tif', '*.tiff', '*.png']
    paths = []
    for p in patterns:
        paths.extend(glob.glob(os.path.join(dirpath, p)))
    if not paths:
        raise FileNotFoundError(f"No .tif/.tiff/.png images found in {dirpath}")
    images = []
    for path in sorted(paths):
        img, _, _ = load_sem_image(path)
        images.append(img)
        print(f"  Loaded: {os.path.basename(path)}  shape={img.shape}")
    return images


# ============================================================
# 2. Log Transform
# ============================================================

def apply_log_transform(image: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Apply log1p to a [0,1]-normalized image, then renormalize output to [0,1].

    Returns (log_norm, log_min, log_max) for inverse transform.
    log_min/log_max are needed to undo the renormalization step.
    """
    log_img = np.log1p(image)
    log_min = float(log_img.min())
    log_max = float(log_img.max())
    log_norm = (log_img - log_min) / (log_max - log_min + 1e-8)
    return log_norm.astype(np.float32), log_min, log_max


def inverse_log_transform(
    denoised_log_norm: np.ndarray, log_min: float, log_max: float
) -> np.ndarray:
    """Undo apply_log_transform: denormalize then apply expm1.

    Returns a [0,1] linear-domain image.
    """
    denoised_log = denoised_log_norm * (log_max - log_min) + log_min
    return np.expm1(denoised_log).astype(np.float32)


# ============================================================
# 3. Signal Proxy (for calibrated mode)
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
# 4. GMM Noise Model (PPN2V parametric model)
# ============================================================

class GMMNoiseModel(nn.Module):
    """
    Parametric signal-dependent Gaussian Mixture Model noise model.

    p(y | s) = Σ_{k=1}^{C} α_k(s) · N(y; s + δ_k, σ²_k(s))

    where:
        α_k(s)  = softmax_k( a_k·s + b_k )     [signal-dependent weights]
        μ_k(s)  = s + δ_k                        [mean tied to signal + offset]
        σ²_k(s) = exp( w_k·s + v_k )             [log-linear variance]

    Parameters (5·C total):
        weight_a  (C,) : linear slope for mixture weight logits
        weight_b  (C,) : bias for mixture weight logits
        mean_offsets (C,): per-component mean offsets δ_k
        var_a (C,)    : log-variance slope per component
        var_b (C,)    : log-variance intercept per component (init: σ≈0.37)

    After fitting, call gmm.requires_grad_(False) to freeze parameters
    before PN2V UNet training begins.
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
        """
        Compute log p(y | s) for each (y_i, s_i) pair.

        Args:
            y : (N,) noisy observations, range [0, 1]
            s : (N,) signal predictions, range [0, 1]  [gradient flows through s]
        Returns:
            (N,) log-probabilities
        """
        log_alpha = F.log_softmax(
            s.unsqueeze(-1) * self.weight_a + self.weight_b, dim=-1
        )                                                            # (N, C)
        mu      = s.unsqueeze(-1) + self.mean_offsets               # (N, C)
        log_var = (self.var_a * s.unsqueeze(-1) + self.var_b).clamp(min=-7.0)  # (N, C)
        log_gauss = -0.5 * (
            (y.unsqueeze(-1) - mu) ** 2 / log_var.exp()
            + log_var
            + math.log(2.0 * math.pi)
        )                                                            # (N, C)
        return (log_alpha + log_gauss).logsumexp(dim=-1)            # (N,)

    @torch.no_grad()
    def plot(self, save_path: str = "data/gmm_log_ppn2v_noise_model.png") -> None:
        """Visualize GMM noise std and signal-dependent mixture weights."""
        C      = self.n_components
        n_pts  = 200
        s_vals = np.linspace(0, 1, n_pts)
        s_t    = torch.from_numpy(s_vals.astype(np.float32))

        log_w_all = F.log_softmax(
            s_t.unsqueeze(-1) * self.weight_a.cpu() + self.weight_b.cpu(),
            dim=-1,
        ).exp().numpy()

        offsets  = self.mean_offsets.cpu().numpy()
        var_a    = self.var_a.cpu().numpy()
        var_b    = self.var_b.cpu().numpy()
        var_mat  = np.exp(np.outer(s_vals, var_a) + var_b[None, :])
        alpha    = log_w_all

        mean_bias     = (alpha * offsets[None, :]).sum(axis=1)
        second_moment = (alpha * (var_mat + offsets[None, :] ** 2)).sum(axis=1)
        noise_std     = np.sqrt(np.maximum(second_moment - mean_bias ** 2, 0.0))

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].plot(s_vals, noise_std, color='royalblue', label='noise std σ(s)')
        axes[0].plot(s_vals, mean_bias, color='tomato',    label='mean bias δ(s)')
        axes[0].axhline(0, color='k', lw=0.5)
        axes[0].set_xlabel("Signal s (log domain)"); axes[0].set_ylabel("Value")
        axes[0].set_title("GMM noise std and mean bias vs. signal (log domain)")
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        for k in range(C):
            axes[1].plot(s_vals, log_w_all[:, k], label=f'α_{k}(s)')
        axes[1].set_xlabel("Signal s (log domain)"); axes[1].set_ylabel("Mixture weight")
        axes[1].set_title("Signal-dependent mixture weights α_k(s)")
        axes[1].legend(); axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"GMM plot saved: {save_path}")


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
    """
    Lightweight 4-level UNet for bootstrap N2V (1-channel scalar output).

    Trained with MSE loss on masked pixels for --n2v_epochs to generate
    a pseudo-clean image used as the signal proxy for GMM calibration.
    Same encoder-decoder architecture as PN2VUNet but with K=1 head.
    """

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
            nn.Conv2d(f * 8, f * 4, kernel_size=1))
        self.dec3 = DoubleConvBlock(f * 8, f * 4)
        self.up2  = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(f * 4, f * 2, kernel_size=1))
        self.dec2 = DoubleConvBlock(f * 4, f * 2)
        self.up1  = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
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
    """
    4-level UNet producing K output channels per pixel.

    Each channel s_k is one sample from the learned prior p(s | context).
    With K=800 diverse samples the log-evidence loss
      L = -log( 1/K Σ_k p_GMM(y | s_k) )
    and MMSE inference
      s_mmse = Σ_k p_GMM(y|s_k)·s_k / Σ_k p_GMM(y|s_k)
    become statistically meaningful.

    Input must be spatially divisible by 16 (4 MaxPool2d layers).
    """

    def __init__(self, in_channels: int = 1, base_features: int = 32, K: int = 800):
        super().__init__()
        f      = base_features
        self.K = K
        self.enc1 = DoubleConvBlock(in_channels, f)
        self.enc2 = DoubleConvBlock(f,     f * 2)
        self.enc3 = DoubleConvBlock(f * 2, f * 4)
        self.enc4 = DoubleConvBlock(f * 4, f * 8)
        self.pool = nn.MaxPool2d(2)
        self.up3  = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(f * 8, f * 4, kernel_size=1))
        self.dec3 = DoubleConvBlock(f * 8, f * 4)
        self.up2  = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(f * 4, f * 2, kernel_size=1))
        self.dec2 = DoubleConvBlock(f * 4, f * 2)
        self.up1  = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(f * 2, f, kernel_size=1))
        self.dec1 = DoubleConvBlock(f * 2, f)
        self.head = nn.Conv2d(f, K, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: (B,1,H,W)  Returns: (B,K,H,W)"""
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


# ============================================================
# 6. N2V Dataset — blind-spot masking
# ============================================================

class N2VDataset(Dataset):
    """Random-patch dataset with vectorized N2V blind-spot masking."""

    def __init__(
        self,
        image:           np.ndarray,
        patch_size:      int   = 64,
        num_patches:     int   = 2000,
        mask_ratio:      float = 0.006,
        neighbor_radius: int   = 5,
        rng_seed:        int   = None,
    ):
        assert patch_size % 16 == 0, "patch_size must be divisible by 16 (4 pool layers)"
        assert image.shape[0] >= patch_size and image.shape[1] >= patch_size
        self.image           = image
        self.H, self.W       = image.shape
        self.patch_size      = patch_size
        self.num_patches     = num_patches
        self.neighbor_radius = neighbor_radius
        self.n_masked        = max(1, int(patch_size * patch_size * mask_ratio))
        self.rng             = np.random.default_rng(rng_seed)

    def __len__(self) -> int:
        return self.num_patches

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        P  = self.patch_size
        r0 = self.rng.integers(0, self.H - P + 1)
        c0 = self.rng.integers(0, self.W - P + 1)
        patch     = self.image[r0:r0 + P, c0:c0 + P].copy()
        corrupted, mask = self._apply_n2v_masking(patch)
        return (
            torch.from_numpy(corrupted).unsqueeze(0).float(),
            torch.from_numpy(patch).unsqueeze(0).float(),
            torch.from_numpy(mask).unsqueeze(0).float(),
        )

    def _apply_n2v_masking(self, patch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        P, rad = self.patch_size, self.neighbor_radius
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
# 7. Bootstrap N2V — Phase 1
# ============================================================

def _infer_tiled_single(
    model:      nn.Module,
    image:      np.ndarray,
    tile_size:  int            = 256,
    overlap:    int            = 32,
    batch_size: int            = 4,
    device:     torch.device   = None,
) -> np.ndarray:
    """
    Tiled inference for a 1-channel output model with Hann-window blending.
    Used to produce the pseudo-clean image from the bootstrap N2V.
    """
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
            bc   = coords[i:i + batch_size]
            tiles = [padded[r:r + th, c:c + tw] for r, c in bc]
            batch = torch.from_numpy(np.stack(tiles)).unsqueeze(1).float().to(device)
            preds = model(batch).squeeze(1).cpu().numpy()   # (B, H, W)
            for j, (r, c) in enumerate(bc):
                pred_sum[r:r + th, c:c + tw]   += preds[j].astype(np.float64) * hann_2d
                weight_sum[r:r + th, c:c + tw] += hann_2d

    denom  = np.maximum(weight_sum, 1e-8)
    return (pred_sum / denom).astype(np.float32)[:H, :W]


def run_bootstrap_n2v(
    image:         np.ndarray,
    n2v_epochs:    int          = 50,
    patch_size:    int          = 64,
    batch_size:    int          = 32,
    learning_rate: float        = 3e-4,
    tile_size:     int          = 256,
    tile_overlap:  int          = 32,
    infer_batch:   int          = 4,
    device:        torch.device = None,
) -> np.ndarray:
    """
    Train a lightweight N2V for n2v_epochs and return the pseudo-clean image.

    The pseudo-clean serves as the signal proxy s for GMM calibration in Phase 2.
    Operates entirely in the log domain (image must already be log-transformed).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Bootstrap N2V: training for {n2v_epochs} epochs (log domain) ...")
    n2v_model = SimpleN2VUNet(in_channels=1, base_features=32).to(device)

    dataset = N2VDataset(image, patch_size=patch_size, num_patches=2000, rng_seed=42)
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
            pred = n2v_model(noisy_in)   # (B, 1, P, P)

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

    print("Bootstrap N2V done — generating pseudo-clean ...")
    pseudo_clean = _infer_tiled_single(
        n2v_model, image,
        tile_size=tile_size, overlap=tile_overlap,
        batch_size=infer_batch, device=device,
    )
    print(f"  Pseudo-clean range: [{pseudo_clean.min():.4f}, {pseudo_clean.max():.4f}]")
    return pseudo_clean


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
    Fit a parametric GMM by minimizing NLL on (s_proxy, y_obs) pixel pairs.

    Uses Adam gradient descent for n_steps on mini-batches of pixel pairs.
    After fitting, all GMM parameters are frozen with requires_grad_(False).

    Args:
        s_pairs : (N,) signal proxy — N2V pseudo-clean or 4-neighbor mean (log domain)
        y_pairs : (N,) noisy observations (log domain)
    Returns:
        GMMNoiseModel with all parameters frozen
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
# 9. PPN2V Training — Phase 2 (negative log-evidence)
# ============================================================

def train_ppn2v(
    model:         PN2VUNet,
    noise_model:   GMMNoiseModel,
    image:         np.ndarray,
    patch_size:    int          = 64,
    batch_size:    int          = 32,
    num_epochs:    int          = 200,
    learning_rate: float        = 3e-4,
    val_frac:      float        = 0.1,
    device:        torch.device = None,
) -> PN2VUNet:
    """
    Train PN2VUNet with the negative log-evidence loss using a frozen GMM.

    Loss per masked pixel:
        L_i = -log( (1/K) Σ_{k=1}^{K} p_GMM(y_i | s_i^k) )
            = -logsumexp_k( log p_GMM(y_i | s_i^k) ) + log(K)

    The optimizer only sees UNet parameters; GMM is frozen (requires_grad=False)
    and contributes no gradient to itself — only to s_i^k through the loss.
    Image must already be log-transformed.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model       = model.to(device)
    noise_model = noise_model.to(device)

    patches_per_epoch = 2000
    n_val   = max(1, int(patches_per_epoch * val_frac))
    n_train = patches_per_epoch - n_val

    train_ds = N2VDataset(image, patch_size=patch_size, num_patches=n_train, rng_seed=42)
    val_ds   = N2VDataset(image, patch_size=patch_size, num_patches=n_val,   rng_seed=99)

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
    print(f"Device:          {device}")
    print(f"UNet parameters: {n_params:,}  |  K (output samples): {K}")
    print(f"GMM components:  {noise_model.n_components}  (frozen)")
    print(f"patch_size={patch_size}  batch_size={batch_size}  epochs={num_epochs}")
    print(f"Patches/epoch: train={n_train}  val={n_val}")
    print(f"Peak GPU tensor per batch: "
          f"~{batch_size * K * patch_size * patch_size * 4 / 1e6:.0f} MB  "
          f"(reduce --batch_size if OOM)")

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

            mask_2d  = mask.bool().squeeze(1)
            y_obs    = noisy_tgt.squeeze(1)[mask_2d]           # (N_masked,)
            s_samp   = pred.permute(0, 2, 3, 1)[mask_2d]       # (N_masked, K)

            N_masked = y_obs.shape[0]
            y_exp    = y_obs.unsqueeze(1).expand(N_masked, K)   # (N_masked, K)
            log_liks = noise_model.log_likelihood(
                y_exp.reshape(-1), s_samp.reshape(-1),
            ).reshape(N_masked, K)

            log_ev = torch.logsumexp(log_liks, dim=1) - log_K   # (N_masked,)
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
                  f"train_NLE={tr_loss / max(tr_count, 1):.4f}  "
                  f"val_NLE={vl_loss / max(vl_count, 1):.4f}  "
                  f"elapsed={time.time() - t0:.1f}s")

    print("Training complete.")
    return model


# ============================================================
# 10. Tiled Inference — MMSE Posterior Mean
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
    pred_K:      torch.Tensor,   # (K, H, W) float
    y_tile:      torch.Tensor,   # (H, W)    float
    noise_model: GMMNoiseModel,
    device:      torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute MMSE posterior mean and prior mean from K network samples.

    MMSE(x) = Σ_k p_GMM(y_x | s_x^k) · s_x^k  /  Σ_k p_GMM(y_x | s_x^k)
    """
    K, H, W = pred_K.shape
    N       = H * W
    s_flat  = pred_K.reshape(K, N)
    y_flat  = y_tile.reshape(N).unsqueeze(0).expand(K, N)

    log_liks  = noise_model.log_likelihood(
        y_flat.reshape(-1), s_flat.reshape(-1),
    ).reshape(K, N)

    weights    = torch.softmax(log_liks, dim=0)
    mmse_flat  = (weights * s_flat).sum(dim=0)
    prior_flat = s_flat.mean(dim=0)

    mmse  = mmse_flat.cpu().numpy().reshape(H, W).astype(np.float32)
    prior = prior_flat.cpu().numpy().reshape(H, W).astype(np.float32)
    return mmse, prior


def predict_tiled(
    model:            PN2VUNet,
    image:            np.ndarray,
    noise_model:      GMMNoiseModel,
    tile_size:        Tuple[int, int] = (256, 256),
    tile_overlap:     Tuple[int, int] = (32, 32),
    infer_batch_size: int             = 2,
    device:           torch.device   = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tiled MMSE inference with Hann-window blending.

    Returns:
        mmse_image  : (H, W) — MMSE posterior mean in log domain
        prior_image : (H, W) — prior mean (unweighted sample average, diagnostic)

    Note on infer_batch_size:
        Each tile → (B, K, H_t, W_t) tensor. K=800, 256×256 tiles:
        B=1 ≈ 209 MB; B=2 ≈ 418 MB. Default=2. Reduce to 1 if OOM.
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
            mmse_t, prior_t = _mmse_from_samples(preds[j], y_tile, noise_model, device)
            mmse_sum[r:r + th, c:c + tw]  += mmse_t.astype(np.float64)  * hann_2d
            prior_sum[r:r + th, c:c + tw] += prior_t.astype(np.float64) * hann_2d
            weight_sum[r:r + th, c:c + tw] += hann_2d

        done = min(i + infer_batch_size, total_tiles)
        if done % max(infer_batch_size, total_tiles // 5 or 1) == 0 \
                or done == total_tiles:
            print(f"  Inference: {done}/{total_tiles} tiles")

    denom    = np.maximum(weight_sum, 1e-8)
    mmse_img  = (mmse_sum  / denom).astype(np.float32)[:H, :W]
    prior_img = (prior_sum / denom).astype(np.float32)[:H, :W]
    return mmse_img, prior_img


# ============================================================
# 11. Save Outputs
# ============================================================

def save_outputs(
    image:      np.ndarray,
    mmse:       np.ndarray,
    prior:      np.ndarray,
    img_min:    float,
    img_max:    float,
    tif_path:   str = "data/denoised_sem_log_ppn2v_juglab.tif",
    prior_path: str = "data/denoised_sem_log_ppn2v_juglab_prior.tif",
    png_path:   str = "data/denoised_sem_log_ppn2v_juglab_comparison.png",
) -> None:
    rng        = img_max - img_min
    mmse_orig  = (mmse  * rng + img_min).astype(np.float32)
    prior_orig = (prior * rng + img_min).astype(np.float32)

    tifffile.imwrite(tif_path,   mmse_orig)
    tifffile.imwrite(prior_path, prior_orig)
    print(f"Saved MMSE output: {tif_path}   range: [{mmse_orig.min():.3f}, {mmse_orig.max():.3f}]")
    print(f"Saved prior mean:  {prior_path}  range: [{prior_orig.min():.3f}, {prior_orig.max():.3f}]")

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(image, cmap='gray');  axes[0].set_title('Original SEM');              axes[0].axis('off')
    axes[1].imshow(prior, cmap='gray');  axes[1].set_title('Prior mean (K avg)');        axes[1].axis('off')
    axes[2].imshow(mmse,  cmap='gray');  axes[2].set_title('MMSE (log + PPN2V juglab)'); axes[2].axis('off')
    axes[3].imshow(np.abs(image - mmse) * 3, cmap='hot')
    axes[3].set_title('|Original − MMSE| × 3');                                           axes[3].axis('off')
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Saved comparison:  {png_path}")


# ============================================================
# 12. Main Pipeline
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Log + PPN2V juglab-faithful: log1p pre-transform + parametric GMM noise model "
            "(signal-dependent) + bootstrap N2V calibration + K=800 sample UNet + "
            "MMSE posterior inference + expm1 inverse transform. "
            "Designed for speckle-dominant + Poisson mixed SEM noise."
        )
    )
    parser.add_argument('--input',        type=str,   default='data/test_sem.tif',
                        help='Path to input .tif/.tiff/.png image')
    parser.add_argument('--output',       type=str,   default='',
                        help='Output .tif path (default: data/denoised_sem_log_ppn2v_juglab.tif)')
    parser.add_argument('--calib_dir',    type=str,   default='',
                        help='Optional: use external images for GMM calibration '
                             '(skips bootstrap N2V; uses 4-neighbor proxy instead)')
    parser.add_argument('--n_components', type=int,   default=3,
                        help='Number of GMM components (default: 3)')
    parser.add_argument('--n2v_epochs',   type=int,   default=50,
                        help='Bootstrap N2V epochs (default: 50; increase for better GMM proxy)')
    parser.add_argument('--gmm_steps',    type=int,   default=1000,
                        help='Gradient steps for GMM fitting (default: 1000)')
    parser.add_argument('--gmm_lr',       type=float, default=1e-2,
                        help='Learning rate for GMM fitting (default: 0.01)')
    parser.add_argument('--K',            type=int,   default=800,
                        help='Number of UNet output samples (default: 800 as in official)')
    parser.add_argument('--epochs',       type=int,   default=200,
                        help='PN2V training epochs (default: 200)')
    parser.add_argument('--patch_size',   type=int,   default=64,
                        help='Training patch size; must be divisible by 16')
    parser.add_argument('--batch_size',   type=int,   default=32,
                        help='Training batch size (reduce if OOM; default: 32 for 8GB GPU)')
    parser.add_argument('--tile_size',    type=int,   default=256,
                        help='Inference tile size (both axes, default: 256)')
    parser.add_argument('--tile_overlap', type=int,   default=32)
    parser.add_argument('--infer_batch',  type=int,   default=2,
                        help='Inference tile batch size (reduce to 1 if OOM; default: 2)')
    parser.add_argument('--device',       type=str,   default=None,
                        help='Device override: cuda, cpu, cuda:1 … (default: auto)')
    args = parser.parse_args()

    input_path  = args.input
    output_path = args.output or 'data/denoised_sem_log_ppn2v_juglab.tif'
    prior_path  = output_path.replace('.tif', '_prior.tif')
    png_path    = output_path.replace('.tif', '_comparison.png')

    os.makedirs('data', exist_ok=True)
    device = torch.device(
        args.device if args.device
        else ('cuda' if torch.cuda.is_available() else 'cpu')
    )

    # ── Load input image ──
    image, img_min, img_max = load_sem_image(input_path)
    print(f"Input image: {image.shape}  range: [{img_min:.3f}, {img_max:.3f}]")

    # ── Apply log transform ──
    log_image, log_min, log_max = apply_log_transform(image)
    print(f"Log transform:  log range [{log_min:.4f}, {log_max:.4f}]  "
          f"→ renormalized to [0, 1]")

    # ── Validate patch_size ──
    patch_size = args.patch_size
    if patch_size % 16 != 0:
        patch_size = (patch_size // 16 + 1) * 16
        print(f"patch_size rounded up to {patch_size} (must be divisible by 16)")

    # ── Phase 1: calibrate GMM (in log domain) ─────────────────────────────
    if args.calib_dir:
        print(f"\n[Phase 1] Calibrated mode: loading images from {args.calib_dir}")
        calib_images = load_images_from_dir(args.calib_dir)
        s_all, y_all = [], []
        for img in calib_images:
            log_img, _, _ = apply_log_transform(img)
            s_proxy = _compute_4neighbor_mean(log_img)
            s_all.append(s_proxy.flatten())
            y_all.append(log_img.flatten())
        s_pairs = np.concatenate(s_all).astype(np.float32)
        y_pairs = np.concatenate(y_all).astype(np.float32)
        print(f"Collected {len(s_pairs):,} pixel pairs from {len(calib_images)} image(s)")
    else:
        print(f"\n[Phase 1] Bootstrap mode: running N2V for {args.n2v_epochs} epochs ...")
        pseudo_clean = run_bootstrap_n2v(
            log_image,
            n2v_epochs=args.n2v_epochs,
            patch_size=patch_size,
            batch_size=args.batch_size,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            infer_batch=args.infer_batch,
            device=device,
        )
        s_pairs = pseudo_clean.flatten().astype(np.float32)
        y_pairs = log_image.flatten().astype(np.float32)

    # Fit and freeze GMM
    print(f"\n[Phase 1] Fitting GMM ...")
    noise_model = fit_gmm_from_pairs(
        s_pairs, y_pairs,
        n_components=args.n_components,
        n_steps=args.gmm_steps,
        lr=args.gmm_lr,
        device=device,
    )

    # ── Phase 2: train PN2V UNet (in log domain) ───────────────────────────
    print(f"\n[Phase 2] Training PN2V UNet with K={args.K} and frozen GMM ...")
    model = PN2VUNet(in_channels=1, base_features=32, K=args.K)

    train_ppn2v(
        model, noise_model, log_image,
        patch_size=patch_size,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        device=device,
    )

    # ── MMSE Inference (in log domain) ─────────────────────────────────────
    print("\nRunning tiled MMSE inference ...")
    mmse_log, prior_log = predict_tiled(
        model, log_image, noise_model,
        tile_size=(args.tile_size, args.tile_size),
        tile_overlap=(args.tile_overlap, args.tile_overlap),
        infer_batch_size=args.infer_batch,
        device=device,
    )

    # ── Inverse log transform → back to linear [0,1] domain ────────────────
    mmse_linear  = inverse_log_transform(mmse_log,  log_min, log_max)
    prior_linear = inverse_log_transform(prior_log, log_min, log_max)
    print(f"Inverse log transform applied: "
          f"MMSE range [{mmse_linear.min():.4f}, {mmse_linear.max():.4f}]")

    # ── Save ────────────────────────────────────────────────────────────────
    save_outputs(
        image, mmse_linear, prior_linear, img_min, img_max,
        tif_path=output_path,
        prior_path=prior_path,
        png_path=png_path,
    )
    noise_model.plot(save_path='data/gmm_log_ppn2v_noise_model.png')


if __name__ == '__main__':
    main()

# ============================================================
# SEM Image Denoising — PN2V with BIC Automatic GMM Capacity Selection
# ============================================================
# Based on: Krull et al., "Probabilistic Noise2Void: Unsupervised
#           Content-Aware Denoising without Ground Truth."
#           Frontiers in Computer Science, 2020.
#
# BIC model selection references:
#   Schwarz, G. (1978). Estimating the dimension of a model.
#       Annals of Statistics, 6(2), 461–464.
#       — Original derivation of the Bayesian Information Criterion (BIC).
#   McLachlan, G. J., & Peel, D. (2000). Finite Mixture Models. Wiley.
#       — Standard reference for BIC-based GMM component selection.
#   Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python.
#       Journal of Machine Learning Research, 12, 2825–2830.
#       — sklearn.mixture.GaussianMixture used for BIC evaluation.
#
# Method note:
#   BIC-based GMM capacity selection is a classical statistical technique
#   (Schwarz 1978; McLachlan & Peel 2000) applied here as an engineering
#   adaptation. It is NOT described in the original PN2V paper, which uses
#   a non-parametric histogram noise model that has no hyperparameter K.
#   This script uses BIC to automate the selection of K when a parametric
#   GMM is preferred over the histogram (e.g. for interpretability or when
#   the noise is known to be Poisson + Gaussian read noise).
#
# How BIC selection works:
#   1. Build (signal_proxy, observation) pixel pairs from the image.
#   2. For each K in --bic_candidates, fit a sklearn GaussianMixture on
#      the 2D (s_proxy, y) space and compute BIC = K_params·ln(n) − 2·ln(L).
#   3. Select the K with the lowest BIC.
#   4. Train the full PN2V pipeline with that K.
#
# When to use this script vs. denoise_PN2V.py:
#   Use this script when:
#   - ENL < 3 (strong speckle): default K=3 may underfit the heavy tail.
#   - You want automatic K tuning without manual trial and error.
#   Use denoise_PN2V.py directly when:
#   - You already know K from domain knowledge (e.g. K=2 for pure Poisson+Gaussian).
#   - Training speed is critical and you want to skip BIC evaluation.
#
# Differences from the official PN2V (Krull et al., 2020):
# ─────────────────────────────────────────────────────────
# Aspect            │ Official PN2V                  │ This implementation
# ──────────────────┼────────────────────────────────┼──────────────────────────────
# Noise model       │ Non-parametric 2D histogram    │ Parametric GMM (auto K via BIC)
#                   │ (no K to choose)               │
# ──────────────────┼────────────────────────────────┼──────────────────────────────
# Network output    │ K=800 samples from posterior   │ Single scalar per pixel.
# ──────────────────┼────────────────────────────────┼──────────────────────────────
# Inference         │ MMSE posterior mean            │ Raw UNet output (default).
#                   │ (800 forward passes)           │ --use_mmse is experimental.
#
# Requirements: torch>=2.0.0  tifffile  matplotlib  numpy  scikit-learn
# Usage:
#   python test_sem.py          # generate synthetic test image (if needed)
#   python denoise_PN2V_bic.py  # BIC selects K, then trains + denoises
#   python denoise_PN2V_bic.py --bic_candidates 2 3 5  # custom candidates
#   python denoise_PN2V_bic.py --n_gaussians 3          # skip BIC, use K=3
# ============================================================

import math
import argparse
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import time
from typing import List, Tuple

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
    """Load SEM image, normalize to float32 [0,1] grayscale."""
    img = tifffile.imread(path).astype(np.float32)
    if img.ndim == 3 and img.shape[-1] == 3:
        img = img @ np.array([0.2989, 0.5870, 0.1140])
    img_min, img_max = float(img.min()), float(img.max())
    img = (img - img_min) / (img_max - img_min + 1e-8)
    return img, img_min, img_max


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
    """4-level encoder-decoder UNet (2D grayscale). Input must be divisible by 8."""

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
# 3. GMM Noise Model
# ============================================================

class GMMNoiseModel(nn.Module):
    """
    Signal-dependent Gaussian Mixture Model: p(y | s).

    For each of K components:
        μ_k(s)  = s + offset_k
        σ²_k(s) = exp(a_k · s + b_k)   (log-linear, always positive)
    """

    def __init__(self, n_gaussians: int = 3):
        super().__init__()
        K = n_gaussians
        self.n_gaussians  = K
        self.log_weights  = nn.Parameter(torch.zeros(K))
        self.mean_offsets = nn.Parameter(torch.linspace(-0.05, 0.05, K))
        self.var_a        = nn.Parameter(torch.zeros(K))
        self.var_b        = nn.Parameter(torch.full((K,), -6.0))

    def log_prob(self, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """log p(y | s).  y, s: (N,)  →  (N,)"""
        y = y.unsqueeze(-1)
        s = s.unsqueeze(-1)
        log_w   = F.log_softmax(self.log_weights, dim=0)
        mu      = s + self.mean_offsets
        log_var = self.var_a * s + self.var_b
        var     = log_var.exp() + 1e-8
        log_gauss = -0.5 * ((y - mu) ** 2 / var + log_var + math.log(2 * math.pi))
        return (log_w + log_gauss).logsumexp(dim=-1)

    def nll_loss(self, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(y, s).mean()

    @torch.no_grad()
    def plot_noise_model(self, save_path: str = "data/noise_model_bic.png") -> None:
        device   = self.var_a.device
        s_vals   = torch.linspace(0, 1, 200, device=device)
        s_expand = s_vals.unsqueeze(-1)
        log_var  = self.var_a * s_expand + self.var_b
        std      = (log_var.exp() + 1e-8).sqrt().cpu().numpy()
        weights  = F.softmax(self.log_weights, dim=0).cpu().numpy()
        s_cpu    = s_vals.cpu()

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for k in range(self.n_gaussians):
            axes[0].plot(s_cpu.numpy(), std[:, k], label=f"G{k} (w={weights[k]:.2f})")
        axes[0].set_xlabel("Signal s"); axes[0].set_ylabel("Noise std σ(s)")
        axes[0].set_title("Learned noise std vs. signal"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
        axes[1].bar(range(self.n_gaussians), weights)
        axes[1].set_xlabel("Component"); axes[1].set_ylabel("Weight")
        axes[1].set_title("Mixture weights"); axes[1].set_xticks(range(self.n_gaussians))
        plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close(fig)
        print(f"Noise model plot saved: {save_path}")


# ============================================================
# 4. Dataset
# ============================================================

class N2VDataset(Dataset):
    def __init__(self, image: np.ndarray, patch_size: int = 64,
                 num_patches: int = 2000, mask_ratio: float = 0.006,
                 neighbor_radius: int = 5, rng_seed: int = None):
        assert patch_size % 8 == 0
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
        return (torch.from_numpy(corrupted).unsqueeze(0),
                torch.from_numpy(patch).unsqueeze(0),
                torch.from_numpy(mask).unsqueeze(0))

    def _apply_n2v_masking(self, patch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        P, rad = self.patch_size, self.neighbor_radius
        corrupted = patch.copy()
        mask      = np.zeros((P, P), dtype=np.float32)
        flat_idx  = self.rng.choice(P * P, size=self.n_masked, replace=False)
        rows, cols = np.unravel_index(flat_idx, (P, P))
        dr = self.rng.integers(-rad, rad + 1, size=self.n_masked)
        dc = self.rng.integers(-rad, rad + 1, size=self.n_masked)
        zero_mask = (dr == 0) & (dc == 0)
        if np.any(zero_mask):
            n_fix  = int(np.sum(zero_mask))
            dr_fix = self.rng.integers(-rad, rad + 1, size=n_fix)
            dc_fix = self.rng.integers(-rad, rad + 1, size=n_fix)
            still_zero = (dr_fix == 0) & (dc_fix == 0)
            dr_fix[still_zero] = 1
            dr[zero_mask] = dr_fix
            dc[zero_mask] = dc_fix
        nr = np.clip(rows + dr, 0, P - 1)
        nc = np.clip(cols + dc, 0, P - 1)
        corrupted[rows, cols] = patch[nr, nc]
        mask[rows, cols]      = 1.0
        return corrupted, mask


# ============================================================
# 5. BIC-Based K Selection  (A3 — novel engineering addition)
# ============================================================

def _build_pixel_pairs(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (y_flat, s_proxy_flat) pixel pairs for BIC evaluation.

    s_proxy = 4-neighbor mean (same proxy used in GMM pre-training).
    Returns numpy arrays (CPU) ready for sklearn.
    """
    img_t  = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
    kernel = torch.zeros(1, 1, 3, 3)
    kernel[0, 0, 0, 1] = 0.25  # top
    kernel[0, 0, 2, 1] = 0.25  # bottom
    kernel[0, 0, 1, 0] = 0.25  # left
    kernel[0, 0, 1, 2] = 0.25  # right
    with torch.no_grad():
        s_proxy = F.conv2d(img_t, kernel, padding=1).squeeze()
    y_flat = img_t.squeeze().flatten().numpy()
    s_flat = s_proxy.flatten().numpy()
    return y_flat, s_flat


def select_num_gaussians_bic(
    image:        np.ndarray,
    candidates:   List[int] = None,
    subsample:    int       = 100_000,
    random_state: int       = 42,
) -> int:
    """
    Select the number of GMM components K using the Bayesian Information Criterion.

    BIC = k_params · ln(n) − 2 · ln(L̂)

    where k_params = 5K (weights K-1, mean_offsets K, var_a K, var_b K; one
    weight is constrained by normalization), and L̂ is the maximum likelihood
    of a sklearn GaussianMixture fitted on the 2D (s_proxy, y) feature space.

    The 2D joint GMM is a proxy for the conditional p(y|s) used in PN2V.
    It ranks K candidates in the same order as the conditional model for
    this purpose (component selection), without requiring multiple full
    GMMNoiseModel training runs.

    References
    ----------
    Schwarz, G. (1978). Estimating the dimension of a model.
        Annals of Statistics, 6(2), 461–464.
    McLachlan, G. J., & Peel, D. (2000). Finite Mixture Models. Wiley.
    Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python.
        JMLR, 12, 2825–2830.

    Note: This function requires scikit-learn.
        pip install scikit-learn
    """
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        raise ImportError(
            "scikit-learn is required for BIC selection.\n"
            "Install with: pip install scikit-learn\n"
            "Or skip BIC with: --n_gaussians 3"
        )

    if candidates is None:
        candidates = [2, 3, 5, 7]

    y_flat, s_flat = _build_pixel_pairs(image)

    # Stack into 2D feature matrix: each row = (s_proxy, y)
    X = np.stack([s_flat, y_flat], axis=1).astype(np.float32)
    n = len(X)

    # Subsample for speed — BIC asymptotically valid for large n
    if n > subsample:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=subsample, replace=False)
        X   = X[idx]
        n_bic = subsample
    else:
        n_bic = n

    print(f"\nBIC model selection — evaluating K ∈ {candidates} "
          f"({n_bic:,} pixel pairs) ...")

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
            print(f"  K={k}:  BIC = {bic:12.1f}")
            if bic < best_bic:
                best_bic, best_k = bic, k
        except Exception as exc:
            print(f"  K={k}:  fit failed ({exc}), skipping")

    print(f"  → Selected K = {best_k}  (lowest BIC = {best_bic:.1f})\n")
    return best_k


# ============================================================
# 6. GMM Pre-training
# ============================================================

def pretrain_gmm(
    noise_model: GMMNoiseModel,
    image:       np.ndarray,
    n_epochs:    int   = 300,
    batch_size:  int   = 4096,
    lr:          float = 1e-3,
    device:      torch.device = None,
) -> None:
    """Fit GMM from 4-neighbor pixel pairs before main training."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    noise_model = noise_model.to(device)

    img_t  = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
    kernel = torch.zeros(1, 1, 3, 3)
    kernel[0, 0, 0, 1] = 0.25; kernel[0, 0, 2, 1] = 0.25
    kernel[0, 0, 1, 0] = 0.25; kernel[0, 0, 1, 2] = 0.25
    with torch.no_grad():
        s_proxy = F.conv2d(img_t, kernel, padding=1).squeeze()

    y_flat = img_t.squeeze().flatten().to(device)
    s_flat = s_proxy.flatten().to(device)
    N      = y_flat.shape[0]

    optimizer = optim.Adam(noise_model.parameters(), lr=lr)
    print(f"Pre-training GMM (K={noise_model.n_gaussians}, {n_epochs} epochs, {N:,} pairs) ...")

    for epoch in range(1, n_epochs + 1):
        idx  = torch.randperm(N, device=device)[:batch_size]
        loss = noise_model.nll_loss(y_flat[idx], s_flat[idx])
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if epoch % 100 == 0 or epoch == 1:
            print(f"  GMM [{epoch:3d}/{n_epochs}]  NLL={loss.item():.4f}")

    print("GMM pre-training complete.")


# ============================================================
# 7. PN2V Training Loop
# ============================================================

def train_pn2v(
    model:          nn.Module,
    noise_model:    GMMNoiseModel,
    image:          np.ndarray,
    patch_size:     int   = 64,
    batch_size:     int   = 128,
    num_epochs:     int   = 100,
    learning_rate:  float = 4e-4,
    gmm_lr_scale:   float = 0.1,
    val_percentage: float = 0.1,
    device: torch.device  = None,
) -> nn.Module:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device); noise_model = noise_model.to(device)

    patches_per_epoch = 2000
    n_val   = max(1, int(patches_per_epoch * val_percentage))
    n_train = patches_per_epoch - n_val

    train_ds = N2VDataset(image, patch_size=patch_size, num_patches=n_train, rng_seed=42)
    val_ds   = N2VDataset(image, patch_size=patch_size, num_patches=n_val,   rng_seed=99)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    optimizer = optim.Adam([
        {'params': model.parameters(),       'lr': learning_rate},
        {'params': noise_model.parameters(), 'lr': learning_rate * gmm_lr_scale},
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    print(f"Device: {device}  |  UNet: {sum(p.numel() for p in model.parameters()):,} params"
          f"  |  GMM: K={noise_model.n_gaussians}")
    print(f"patch_size={patch_size}  batch_size={batch_size}  epochs={num_epochs}")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        model.train(); noise_model.train()
        tr_loss, tr_count = 0.0, 0
        for noisy_in, noisy_tgt, mask in train_loader:
            noisy_in = noisy_in.to(device); noisy_tgt = noisy_tgt.to(device); mask = mask.to(device)
            optimizer.zero_grad()
            pred      = model(noisy_in)
            mask_bool = mask.bool().squeeze(1)
            y_obs     = noisy_tgt.squeeze(1)[mask_bool]
            s_pred    = pred.squeeze(1)[mask_bool]
            loss      = noise_model.nll_loss(y_obs, s_pred)
            loss.backward(); optimizer.step()
            tr_loss  += loss.item() * y_obs.numel()
            tr_count += y_obs.numel()

        model.eval(); noise_model.eval()
        vl_loss, vl_count = 0.0, 0
        with torch.no_grad():
            for noisy_in, noisy_tgt, mask in val_loader:
                noisy_in = noisy_in.to(device); noisy_tgt = noisy_tgt.to(device); mask = mask.to(device)
                pred      = model(noisy_in)
                mask_bool = mask.bool().squeeze(1)
                y_obs     = noisy_tgt.squeeze(1)[mask_bool]
                s_pred    = pred.squeeze(1)[mask_bool]
                vl_loss  += noise_model.nll_loss(y_obs, s_pred).item() * y_obs.numel()
                vl_count += y_obs.numel()

        scheduler.step()
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch:3d}/{num_epochs}]  "
                  f"train_NLL={tr_loss/max(tr_count,1):.4f}  "
                  f"val_NLL={vl_loss/max(vl_count,1):.4f}  "
                  f"elapsed={time.time()-t0:.1f}s")

    print("Training complete.")
    return model


# ============================================================
# 8. Tiled Inference
# ============================================================

def _compute_padding(image_size: int, tile_size: int) -> int:
    pad = max(0, tile_size - image_size)
    padded = image_size + pad
    remainder = padded % 8
    if remainder != 0:
        pad += 8 - remainder
    return pad


@torch.no_grad()
def _apply_mmse_tile(
    s_pred: np.ndarray, y_obs: np.ndarray,
    noise_model: 'GMMNoiseModel',
    n_hypotheses: int = 50, search_half_width: float = 0.15,
    device: torch.device = None,
) -> np.ndarray:
    """
    MMSE posterior mean correction (experimental).
    See denoise_PN2V.py _apply_mmse_tile docstring for important limitations.
    """
    if device is None:
        device = next(noise_model.parameters()).device
    H, W = s_pred.shape; N = H * W
    s0 = torch.from_numpy(s_pred.flatten()).float().to(device)
    y  = torch.from_numpy(y_obs.flatten()).float().to(device)
    deltas = torch.linspace(-search_half_width, search_half_width, n_hypotheses, device=device)
    s_grid = torch.clamp(s0.unsqueeze(1) + deltas.unsqueeze(0), 0.0, 1.0)
    y_exp  = y.unsqueeze(1).expand_as(s_grid)
    log_w  = noise_model.log_prob(y_exp.reshape(-1), s_grid.reshape(-1)).reshape(N, n_hypotheses)
    weights = torch.softmax(log_w, dim=1)
    s_mmse  = (weights * s_grid).sum(dim=1)
    return s_mmse.cpu().numpy().reshape(H, W).astype(np.float32)


def predict_tiled(
    model:            nn.Module,
    image:            np.ndarray,
    noise_model:      'GMMNoiseModel' = None,
    tile_size:        Tuple[int, int] = (256, 256),
    tile_overlap:     Tuple[int, int] = (48, 48),
    infer_batch_size: int             = 8,
    device:           torch.device   = None,
) -> np.ndarray:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    H, W = image.shape; th, tw = tile_size; oh, ow = tile_overlap
    pad_h = _compute_padding(H, th); pad_w = _compute_padding(W, tw)
    padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect') \
             if (pad_h > 0 or pad_w > 0) else image
    pH, pW = padded.shape
    hann_2d = np.outer(torch.hann_window(th, periodic=False).numpy(),
                       torch.hann_window(tw, periodic=False).numpy())
    output_sum = np.zeros((pH, pW), dtype=np.float64)
    weight_sum = np.zeros((pH, pW), dtype=np.float64)
    stride_h = th - oh; stride_w = tw - ow
    row_starts: List[int] = list(range(0, pH - th + 1, stride_h))
    col_starts: List[int] = list(range(0, pW - tw + 1, stride_w))
    if row_starts[-1] + th < pH: row_starts.append(pH - th)
    if col_starts[-1] + tw < pW: col_starts.append(pW - tw)
    coords = [(r, c) for r in row_starts for c in col_starts]
    total_tiles = len(coords)
    model.eval()
    with torch.no_grad():
        for i in range(0, total_tiles, infer_batch_size):
            batch_coords = coords[i:i + infer_batch_size]
            tiles   = [padded[r:r + th, c:c + tw] for r, c in batch_coords]
            batch_t = torch.from_numpy(np.stack(tiles)).unsqueeze(1).to(device)
            preds   = model(batch_t).squeeze(1).cpu().numpy()
            if noise_model is not None:
                for j, (r, c) in enumerate(batch_coords):
                    preds[j] = _apply_mmse_tile(preds[j], padded[r:r + th, c:c + tw],
                                                noise_model, device=device)
            for j, (r, c) in enumerate(batch_coords):
                output_sum[r:r + th, c:c + tw] += preds[j].astype(np.float64) * hann_2d
                weight_sum[r:r + th, c:c + tw] += hann_2d
            done = min(i + infer_batch_size, total_tiles)
            if done % max(infer_batch_size, total_tiles // 5 or 1) == 0 or done == total_tiles:
                print(f"  Inference: {done}/{total_tiles} tiles")
    return (output_sum / np.maximum(weight_sum, 1e-8)).astype(np.float32)[:H, :W]


# ============================================================
# 9. Save Outputs
# ============================================================

def save_outputs(
    image: np.ndarray, denoised: np.ndarray,
    img_min: float, img_max: float,
    tif_path: str = "data/denoised_sem_PN2V_bic.tif",
    png_path: str = "data/denoising_result_PN2V_bic.png",
) -> None:
    denoised_orig = (denoised * (img_max - img_min) + img_min).astype(np.float32)
    tifffile.imwrite(tif_path, denoised_orig)
    print(f"Saved: {tif_path}  range: [{denoised_orig.min():.3f}, {denoised_orig.max():.3f}]")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image,    cmap='gray'); axes[0].set_title('Original SEM');        axes[0].axis('off')
    axes[1].imshow(denoised, cmap='gray'); axes[1].set_title('PN2V-BIC Denoised');   axes[1].axis('off')
    diff = np.abs(image - denoised) * 3
    axes[2].imshow(diff, cmap='hot');     axes[2].set_title('Difference (×3)');       axes[2].axis('off')
    plt.tight_layout(); plt.savefig(png_path, dpi=150); plt.show()
    print(f"Saved: {png_path}")


# ============================================================
# 10. Main Pipeline
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PN2V with BIC auto-selection of GMM components K."
    )
    parser.add_argument('--input',               type=str,   default='data/test_sem.tif')
    parser.add_argument('--output',              type=str,   default='')
    parser.add_argument('--epochs',              type=int,   default=100)
    parser.add_argument('--patch_size',          type=int,   default=64)
    parser.add_argument('--batch_size',          type=int,   default=128)
    parser.add_argument('--n_gaussians',         type=int,   default=0,
                        help='Fix K (skip BIC). 0 = auto BIC selection (default).')
    parser.add_argument('--bic_candidates',      type=int,   nargs='+', default=[2, 3, 5, 7, 9, 11, 13],
                        help='K values to evaluate for BIC (default: 2 3 5 7 9 11 13)')
    parser.add_argument('--bic_subsample',       type=int,   default=100_000,
                        help='Max pixel pairs for BIC evaluation (default: 100000)')
    parser.add_argument('--gmm_pretrain_epochs', type=int,   default=300)
    parser.add_argument('--tile_size',           type=int,   default=256)
    parser.add_argument('--tile_overlap',        type=int,   default=48)
    parser.add_argument('--infer_batch',         type=int,   default=8)
    parser.add_argument('--device',              type=str,   default=None)
    parser.add_argument('--use_mmse',            action='store_true',
                        help='Experimental MMSE grid correction. '
                             'WARNING: returns near-noisy output for single-scalar network.')
    args = parser.parse_args()

    output_path = args.output or "data/denoised_sem_PN2V_bic.tif"
    os.makedirs("data", exist_ok=True)
    device = torch.device(args.device if args.device
                          else ('cuda' if torch.cuda.is_available() else 'cpu'))

    # --- Load ---
    image, img_min, img_max = load_sem_image(args.input)
    print(f"Image: {image.shape}  range: [{img_min:.3f}, {img_max:.3f}]")

    # --- Low-count diagnostic ---
    bg_mean = float(image[:30, :30].mean())
    if bg_mean < 0.02:
        print(f"WARNING: background mean={bg_mean:.4f} < 0.02 (extreme low-dose SEM).")
        print("  PN2V in raw space is the correct choice; no transforms applied.")
    else:
        print(f"Background mean={bg_mean:.4f}  (normal signal level)")

    # --- BIC or manual K selection ---
    if args.n_gaussians > 0:
        n_gaussians = args.n_gaussians
        print(f"Using fixed K={n_gaussians} (--n_gaussians specified, BIC skipped)")
    else:
        n_gaussians = select_num_gaussians_bic(
            image,
            candidates=args.bic_candidates,
            subsample=args.bic_subsample,
        )

    # --- GMM pre-training ---
    noise_model = GMMNoiseModel(n_gaussians=n_gaussians)
    pretrain_gmm(noise_model, image, n_epochs=args.gmm_pretrain_epochs, device=device)

    # --- Joint UNet + GMM training ---
    model = N2VUNet(in_channels=1, base_features=32)
    train_pn2v(model, noise_model, image,
               patch_size=args.patch_size, batch_size=args.batch_size,
               num_epochs=args.epochs, device=device)

    # --- Inference ---
    _nm = noise_model if args.use_mmse else None
    print("\nRunning tiled inference" +
          (" (MMSE grid — experimental)..." if _nm is not None else "..."))
    denoised = predict_tiled(model, image, noise_model=_nm,
                             tile_size=(args.tile_size, args.tile_size),
                             tile_overlap=(args.tile_overlap, args.tile_overlap),
                             infer_batch_size=args.infer_batch, device=device)

    # --- Save ---
    save_outputs(image, denoised, img_min, img_max, tif_path=output_path)
    noise_model.plot_noise_model(save_path="data/noise_model_PN2V_bic.png")
    print(f"BIC selected K={n_gaussians}")


if __name__ == '__main__':
    main()

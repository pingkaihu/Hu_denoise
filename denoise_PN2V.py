# ============================================================
# SEM Image Denoising — Probabilistic Noise2Void (pure PyTorch)
# ============================================================
# Based on: Krull et al., "Probabilistic Noise2Void" (2020)
#
# Differences from denoise_N2V.py:
#   + GMMNoiseModel   signal-dependent Gaussian mixture p(y|s)
#   + NLL loss        -log p(y_obs | s_pred) replaces MSE
#   + pretrain_gmm    fits GMM from neighbor pixel pairs before main training
#
# Identical to denoise_N2V.py:
#   = N2VUNet         architecture unchanged
#   = N2VDataset      blind-spot masking unchanged
#   = predict_tiled   batched tiled inference unchanged
#
# Why no GAT pre-processing:
#   The GMM directly models the raw Poisson-Gamma noise distribution.
#   Applying GAT first would require PN2V to model GAT's residual error
#   rather than the original physics, which is mathematically redundant.
#
# Requirements: torch>=2.0.0  tifffile  matplotlib  numpy
# Usage:
#   python test_sem.py      # generate synthetic test image (if needed)
#   python denoise_PN2V.py  # train + denoise -> data/denoised_sem_PN2V.tif
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
# 1. Image Loading  (identical to denoise_N2V.py)
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


# ============================================================
# 2. UNet Architecture  (identical to denoise_N2V.py)
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
# 3. GMM Noise Model  (PN2V addition)
# ============================================================

class GMMNoiseModel(nn.Module):
    """
    Signal-dependent Gaussian Mixture Model for the conditional noise distribution
    p(y | s), where y is the observed noisy pixel and s is the true signal.

    Parameterization
    ----------------
    For each of K components:
        μ_k(s)   = s + offset_k            (signal-centered, learnable bias)
        σ_k²(s)  = exp(a_k · s + b_k)      (log-linear in signal — always positive)
                   a_k > 0  →  Poisson-like (variance grows with signal)
                   a_k = 0  →  constant variance (pure Gaussian read noise)
    Mixture weights are signal-independent (constant softmax over log_weights).

    Why this parameterization:
    - μ_k = s + offset_k reflects that the expected observation is near the
      true signal, with K components capturing multi-modal residuals.
    - log-linear variance naturally handles Poisson shot noise (Var ∝ Mean)
      and Gaussian read noise (Var = const) in a unified differentiable form.
    - All parameters are learnable; gradients flow cleanly through log_prob.

    Choosing K (n_gaussians)
    ------------------------
    K controls the expressive capacity of the noise model. Each component can
    independently learn a different noise regime (e.g. shot noise, read noise,
    outlier spikes). However, more components also increase the risk of
    redundant/collapsed components and slower convergence on a single image.

    Recommended values:
      K = 2 — use when noise is known to be Poisson + Gaussian read noise
              (two physically distinct sources). Lowest risk of collapse.
              Good default for synthetic or well-characterised SEM images.

      K = 3 — (default) suitable for most real SEM scenarios where an
              additional component can absorb residual structure (e.g. mild
              charging artefacts, detector non-linearity). Empirically stable
              when the image has ≥ 256×256 pixels.

      K = 5 — use when noise has visibly multi-modal residuals (e.g. strong
              scan-line modulation on top of shot noise) or when multiple
              physical noise sources are suspected. Requires more training
              data / epochs to avoid degeneracy; not recommended for single
              512×512 images with num_epochs ≤ 100.

    Diagnosing collapse (K too large):
      After training, call noise_model.plot_noise_model(). If two or more
      σ(s) curves are nearly identical, those components have collapsed —
      reduce K by 1 and retrain.  A healthy model shows clearly separated
      curves with diverse slopes (some positive for shot noise, one near-flat
      for read noise).

    Diagnosing underfitting (K too small):
      If train_NLL stops improving early (< 50 epochs) while val_NLL is still
      decreasing, the GMM may lack capacity — try increasing K by 1.
    """

    def __init__(self, n_gaussians: int = 3):
        super().__init__()
        K = n_gaussians
        self.n_gaussians = K

        # Mixture weights (log-space, normalised via softmax during log_prob)
        self.log_weights = nn.Parameter(torch.zeros(K))

        # Mean offsets: spread around 0 to break symmetry at initialisation
        self.mean_offsets = nn.Parameter(torch.linspace(-0.05, 0.05, K))

        # Variance: log σ²_k(s) = a_k · s + b_k
        # Init: a_k=0 (constant), b_k=-6 → σ ≈ 0.025 (reasonable for [0,1] images)
        self.var_a = nn.Parameter(torch.zeros(K))
        self.var_b = nn.Parameter(torch.full((K,), -6.0))

    def log_prob(self, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Compute log p(y | s) under the GMM.

        Args:
            y : observed noisy values, shape (N,)
            s : predicted signal values, shape (N,)
        Returns:
            log p(y | s), shape (N,)
        """
        y = y.unsqueeze(-1)                                    # (N, 1)
        s = s.unsqueeze(-1)                                    # (N, 1)

        log_w   = F.log_softmax(self.log_weights, dim=0)      # (K,)
        mu      = s + self.mean_offsets                        # (N, K)
        log_var = self.var_a * s + self.var_b                  # (N, K)
        var     = log_var.exp() + 1e-8                         # (N, K), strictly positive

        # Log of Gaussian PDF for each component
        log_gauss = -0.5 * (
            (y - mu) ** 2 / var + log_var + math.log(2 * math.pi)
        )                                                      # (N, K)

        # Log-sum-exp for numerical stability
        return (log_w + log_gauss).logsumexp(dim=-1)           # (N,)

    def nll_loss(self, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Mean negative log-likelihood over the batch."""
        return -self.log_prob(y, s).mean()

    @torch.no_grad()
    def plot_noise_model(self, save_path: str = "data/noise_model.png") -> None:
        """
        Visualise learned noise model:
          Left:  predicted noise std as a function of signal level
          Right: estimated mixture weights
        """
        device   = self.var_a.device
        s_vals   = torch.linspace(0, 1, 200, device=device)
        s_expand = s_vals.unsqueeze(-1)                        # (200, 1)

        log_var = self.var_a * s_expand + self.var_b           # (200, K)
        std     = (log_var.exp() + 1e-8).sqrt().cpu().numpy()  # (200, K)
        weights = F.softmax(self.log_weights, dim=0).cpu().numpy()  # (K,)
        s_vals  = s_vals.cpu()

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        for k in range(self.n_gaussians):
            axes[0].plot(s_vals.numpy(), std[:, k],
                         label=f"G{k} (w={weights[k]:.2f})")
        axes[0].set_xlabel("Signal s")
        axes[0].set_ylabel("Noise std σ(s)")
        axes[0].set_title("Learned noise std vs. signal")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].bar(range(self.n_gaussians), weights)
        axes[1].set_xlabel("Gaussian component")
        axes[1].set_ylabel("Mixture weight")
        axes[1].set_title("Mixture weights")
        axes[1].set_xticks(range(self.n_gaussians))

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Noise model plot saved: {save_path}")


# ============================================================
# 4. Dataset  (identical to denoise_N2V.py)
# ============================================================

class N2VDataset(Dataset):
    """Random-patch dataset with vectorized N2V blind-spot masking."""

    def __init__(
        self,
        image: np.ndarray,
        patch_size:      int   = 64,
        num_patches:     int   = 2000,
        mask_ratio:      float = 0.006,
        neighbor_radius: int   = 5,
        rng_seed:        int   = None,
    ):
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
        r0 = self.rng.integers(0, self.H - P)
        c0 = self.rng.integers(0, self.W - P)
        patch     = self.image[r0:r0 + P, c0:c0 + P].copy()
        corrupted, mask = self._apply_n2v_masking(patch)
        return (
            torch.from_numpy(corrupted).unsqueeze(0),   # network input
            torch.from_numpy(patch).unsqueeze(0),        # original noisy (= y_obs)
            torch.from_numpy(mask).unsqueeze(0),         # binary mask
        )

    def _apply_n2v_masking(
        self, patch: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        P, rad = self.patch_size, self.neighbor_radius
        corrupted = patch.copy()
        mask      = np.zeros((P, P), dtype=np.float32)

        flat_idx   = self.rng.choice(P * P, size=self.n_masked, replace=False)
        rows, cols = np.unravel_index(flat_idx, (P, P))

        dr = self.rng.integers(-rad, rad + 1, size=self.n_masked)
        dc = self.rng.integers(-rad, rad + 1, size=self.n_masked)

        zero_mask = (dr == 0) & (dc == 0)
        if np.any(zero_mask):
            n_fix    = int(np.sum(zero_mask))
            shift_dr = self.rng.integers(0, 2, size=n_fix).astype(bool)
            sign     = self.rng.choice([-1, 1], size=n_fix)
            dr[zero_mask] = np.where(shift_dr,  sign, 0)
            dc[zero_mask] = np.where(~shift_dr, sign, 0)

        nr = np.clip(rows + dr, 0, P - 1)
        nc = np.clip(cols + dc, 0, P - 1)
        corrupted[rows, cols] = patch[nr, nc]
        mask[rows, cols]      = 1.0
        return corrupted, mask


# ============================================================
# 5. GMM Pre-training  (PN2V addition)
# ============================================================

def pretrain_gmm(
    noise_model: GMMNoiseModel,
    image:       np.ndarray,
    n_epochs:    int   = 300,
    batch_size:  int   = 4096,
    lr:          float = 1e-3,
    device:      torch.device = None,
) -> None:
    """
    Fit the GMM noise model before main training using neighbor-pixel pairs
    as a proxy for (signal, observation) pairs.

    Strategy
    --------
    For each pixel y[r, c], the 4-neighbor mean is a low-noise proxy for
    the underlying signal s:
        s_proxy[r, c] = mean(y[r-1,c], y[r+1,c], y[r,c-1], y[r,c+1])

    Minimising -log p(y | s_proxy) fits the noise model to the empirical
    relationship between signal level and noise distribution — without any
    paired clean/noisy data.

    Note: s_proxy underestimates noise variance slightly (averaging reduces
    variance), but this is acceptable for initialisation. The GMM is refined
    jointly with the UNet during the main training phase.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    noise_model = noise_model.to(device)

    img = torch.from_numpy(image).float()

    # Build 4-neighbor mean as signal proxy (via average pooling, padding=1)
    img_t = img.unsqueeze(0).unsqueeze(0)                    # (1,1,H,W)
    kernel = torch.ones(1, 1, 3, 3) / 4.0
    # Exclude center: set center weight to 0
    kernel[0, 0, 1, 1] = 0.0
    # Only 4 cardinal neighbors (not corners)
    kernel[0, 0, 0, 0] = 0.0
    kernel[0, 0, 0, 2] = 0.0
    kernel[0, 0, 2, 0] = 0.0
    kernel[0, 0, 2, 2] = 0.0

    with torch.no_grad():
        s_proxy = F.conv2d(img_t, kernel, padding=1).squeeze()  # (H, W)

    y_flat = img.flatten()                 # (N,)
    s_flat = s_proxy.flatten()             # (N,)

    # Move to device
    y_flat = y_flat.to(device)
    s_flat = s_flat.to(device)
    N = y_flat.shape[0]

    optimizer = optim.Adam(noise_model.parameters(), lr=lr)

    print(f"Pre-training GMM ({noise_model.n_gaussians} Gaussians, "
          f"{n_epochs} epochs, {N:,} pixel pairs) ...")

    for epoch in range(1, n_epochs + 1):
        idx   = torch.randperm(N, device=device)[:batch_size]
        y_b   = y_flat[idx]
        s_b   = s_flat[idx]
        loss  = noise_model.nll_loss(y_b, s_b)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0 or epoch == 1:
            print(f"  GMM pre-train [{epoch:3d}/{n_epochs}]  NLL={loss.item():.4f}")

    print("GMM pre-training complete.")


# ============================================================
# 6. PN2V Training Loop  (NLL loss replaces MSE)
# ============================================================

def train_pn2v(
    model:        nn.Module,
    noise_model:  GMMNoiseModel,
    image:        np.ndarray,
    patch_size:     int   = 64,
    batch_size:     int   = 128,
    num_epochs:     int   = 100,
    learning_rate:  float = 4e-4,
    gmm_lr_scale:   float = 0.1,
    val_percentage: float = 0.1,
    device: torch.device  = None,
) -> nn.Module:
    """
    Joint training of UNet + GMMNoiseModel with NLL loss.

    Loss
    ----
    For each masked pixel position i:
        L_i = -log p(y_obs_i | s_pred_i)
    where y_obs_i is the original noisy value and s_pred_i is the UNet output.

    The GMM is trained at a lower learning rate (gmm_lr_scale × lr) to
    prevent it from drifting too far from the pre-trained initialisation
    before the UNet has converged.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model       = model.to(device)
    noise_model = noise_model.to(device)

    patches_per_epoch = 2000
    n_val   = max(1, int(patches_per_epoch * val_percentage))
    n_train = patches_per_epoch - n_val

    train_ds = N2VDataset(image, patch_size=patch_size, num_patches=n_train, rng_seed=42)
    val_ds   = N2VDataset(image, patch_size=patch_size, num_patches=n_val,   rng_seed=99)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0)

    # Separate learning rates: UNet at lr, GMM at lr * gmm_lr_scale
    optimizer = optim.Adam([
        {'params': model.parameters(),       'lr': learning_rate},
        {'params': noise_model.parameters(), 'lr': learning_rate * gmm_lr_scale},
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    n_params_net = sum(p.numel() for p in model.parameters())
    n_params_gmm = sum(p.numel() for p in noise_model.parameters())
    print(f"Device: {device}")
    print(f"UNet parameters: {n_params_net:,}  |  GMM parameters: {n_params_gmm}")
    print(f"Training: patch_size={patch_size}  batch_size={batch_size}  epochs={num_epochs}")
    print(f"Patches/epoch: train={n_train}  val={n_val}")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        model.train()
        noise_model.train()
        tr_loss, tr_count = 0.0, 0

        for noisy_in, noisy_tgt, mask in train_loader:
            noisy_in  = noisy_in.to(device)
            noisy_tgt = noisy_tgt.to(device)
            mask      = mask.to(device)

            optimizer.zero_grad()
            pred = model(noisy_in)                          # (B,1,P,P) predicted signal

            # Extract masked positions only
            mask_bool  = mask.bool().squeeze(1)             # (B, P, P)
            y_obs  = noisy_tgt.squeeze(1)[mask_bool]        # observed noisy (N_masked,)
            s_pred = pred.squeeze(1)[mask_bool]             # predicted signal (N_masked,)

            loss = noise_model.nll_loss(y_obs, s_pred)
            loss.backward()
            optimizer.step()

            tr_loss  += loss.item() * y_obs.numel()
            tr_count += y_obs.numel()

        model.eval()
        noise_model.eval()
        vl_loss, vl_count = 0.0, 0

        with torch.no_grad():
            for noisy_in, noisy_tgt, mask in val_loader:
                noisy_in  = noisy_in.to(device)
                noisy_tgt = noisy_tgt.to(device)
                mask      = mask.to(device)

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
# 7. Tiled Inference  (identical to denoise_N2V.py)
# ============================================================

def _compute_padding(image_size: int, tile_size: int) -> int:
    pad     = max(0, tile_size - image_size)
    padded  = image_size + pad
    remainder = padded % 8
    if remainder != 0:
        pad += 8 - remainder
    return pad


def predict_tiled(
    model:           nn.Module,
    image:           np.ndarray,
    tile_size:       Tuple[int, int] = (256, 256),
    tile_overlap:    Tuple[int, int] = (48, 48),
    infer_batch_size: int            = 8,
    device:          torch.device    = None,
) -> np.ndarray:
    """Batched tiled inference with Hann-window blending and reflection padding."""
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
    output_sum = np.zeros((pH, pW), dtype=np.float64)
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
    with torch.no_grad():
        for i in range(0, total_tiles, infer_batch_size):
            batch_coords = coords[i:i + infer_batch_size]
            tiles   = [padded[r:r + th, c:c + tw] for r, c in batch_coords]
            batch_t = torch.from_numpy(np.stack(tiles)).unsqueeze(1).to(device)
            preds   = model(batch_t).squeeze(1).cpu().numpy()
            for j, (r, c) in enumerate(batch_coords):
                output_sum[r:r + th, c:c + tw] += preds[j].astype(np.float64) * hann_2d
                weight_sum[r:r + th, c:c + tw] += hann_2d

            done = min(i + infer_batch_size, total_tiles)
            if done % max(infer_batch_size, total_tiles // 5 or 1) == 0 \
                    or done == total_tiles:
                print(f"  Inference: {done}/{total_tiles} tiles")

    return (output_sum / np.maximum(weight_sum, 1e-8)).astype(np.float32)[:H, :W]


# ============================================================
# 8. Save Outputs
# ============================================================

def save_outputs(
    image:    np.ndarray,
    denoised: np.ndarray,
    img_min:  float,
    img_max:  float,
    tif_path: str = "data/denoised_sem_PN2V.tif",
    png_path: str = "data/denoising_result_PN2V.png",
) -> None:
    denoised_orig = (denoised * (img_max - img_min) + img_min).astype(np.float32)
    tifffile.imwrite(tif_path, denoised_orig)
    print(f"Saved: {tif_path}  range: [{denoised_orig.min():.3f}, {denoised_orig.max():.3f}]")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image,    cmap='gray'); axes[0].set_title('Original SEM'); axes[0].axis('off')
    axes[1].imshow(denoised, cmap='gray'); axes[1].set_title('PN2V Denoised'); axes[1].axis('off')
    diff = np.abs(image - denoised) * 3
    axes[2].imshow(diff, cmap='hot');     axes[2].set_title('Difference (×3)'); axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.show()
    print(f"Saved: {png_path}")


# ============================================================
# 9. Main Pipeline
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PN2V SEM denoiser: probabilistic N2V with GMM noise model."
    )
    parser.add_argument('--input',              type=str, default='data/test_sem.tif',
                        help='Path to input .tif/.tiff/.png image')
    parser.add_argument('--output',             type=str, default='',
                        help='Path to output .tif (default: data/denoised_sem_PN2V.tif)')
    parser.add_argument('--epochs',             type=int, default=100)
    parser.add_argument('--patch_size',         type=int, default=64)
    parser.add_argument('--batch_size',         type=int, default=128)
    parser.add_argument('--n_gaussians',        type=int, default=3,
                        help='Number of GMM components for noise model')
    parser.add_argument('--gmm_pretrain_epochs',type=int, default=300,
                        help='Epochs for GMM pre-training; reduce for extreme low-dose SEM')
    parser.add_argument('--tile_size',          type=int, default=256,
                        help='Inference tile size applied to both H and W')
    parser.add_argument('--tile_overlap',       type=int, default=48)
    parser.add_argument('--infer_batch',        type=int, default=8)
    parser.add_argument('--device',             type=str, default=None,
                        help='Device override: cuda, cpu, cuda:1 … (default: auto)')
    args = parser.parse_args()

    input_path          = args.input
    output_path         = args.output or "data/denoised_sem_PN2V.tif"
    patch_size          = args.patch_size
    batch_size          = args.batch_size
    num_epochs          = args.epochs
    n_gaussians         = args.n_gaussians
    gmm_pretrain_epochs = args.gmm_pretrain_epochs
    tile_size           = (args.tile_size, args.tile_size)
    tile_overlap        = (args.tile_overlap, args.tile_overlap)
    infer_batch_size    = args.infer_batch

    os.makedirs("data", exist_ok=True)

    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))

    # --- Load ---
    image, img_min, img_max = load_sem_image(input_path)
    print(f"Image: {image.shape}  range: [{img_min:.3f}, {img_max:.3f}]")

    # --- Low-count diagnostic ---
    bg_mean = float(image[:30, :30].mean())
    if bg_mean < 0.02:
        print(f"WARNING: background mean={bg_mean:.4f} < 0.02 (extreme low-dose SEM).")
        print("  Log/GAT transforms would introduce non-linear bias in dark regions.")
        print("  PN2V in raw space is the correct choice here.")
    else:
        print(f"Background mean={bg_mean:.4f}  (normal signal level, no transform needed)")

    # --- GMM pre-training ---
    noise_model = GMMNoiseModel(n_gaussians=n_gaussians)
    pretrain_gmm(noise_model, image, n_epochs=gmm_pretrain_epochs, device=device)

    # --- Joint UNet + GMM training ---
    model = N2VUNet(in_channels=1, base_features=32)
    train_pn2v(
        model, noise_model, image,
        patch_size=patch_size,
        batch_size=batch_size,
        num_epochs=num_epochs,
        device=device,
    )

    # --- Inference ---
    print("\nRunning tiled inference...")
    denoised = predict_tiled(
        model, image,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        infer_batch_size=infer_batch_size,
        device=device,
    )

    # --- Save ---
    save_outputs(image, denoised, img_min, img_max, tif_path=output_path)
    noise_model.plot_noise_model(save_path="data/noise_model_PN2V.png")


if __name__ == '__main__':
    main()

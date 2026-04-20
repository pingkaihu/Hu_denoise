# ============================================================
# SEM Image Denoising — Probabilistic Noise2Void (juglab-faithful)
# ============================================================
# Based on: Krull et al., "Probabilistic Noise2Void" (2020)
#   Frontiers in Computer Science, doi:10.3389/fcomp.2020.00005
#   GitHub: github.com/juglab/pn2v
#
# This script ports the official juglab/pn2v implementation to a
# self-contained local PyTorch script without any external framework.
#
# Key differences from denoise_PN2V.py (parametric GMM variant):
# ─────────────────────────────────────────────────────────────
# Aspect            │ denoise_PN2V.py (this repo)  │ This script (juglab-faithful)
# ──────────────────┼───────────────────────────────┼───────────────────────────────
# Noise model       │ Parametric GMM (K Gaussians)  │ Non-parametric 2D histogram
#                   │ Learned jointly with UNet      │ 256×256 bins, p(y|s) per row
#                   │ GMM pre-training phase needed  │ Built once from pixel pairs
# ──────────────────┼───────────────────────────────┼───────────────────────────────
# UNet output       │ 1 scalar per pixel             │ K=800 samples per pixel
#                   │                                │ (K channels in final head)
# ──────────────────┼───────────────────────────────┼───────────────────────────────
# Training loss     │ GMM NLL: -log p_GMM(y|s)      │ Neg. log-evidence:
#                   │                                │ -log( 1/K Σ_k p_hist(y|s_k) )
# ──────────────────┼───────────────────────────────┼───────────────────────────────
# Inference         │ Raw UNet scalar                │ MMSE posterior mean:
#                   │                                │ Σ_k p(y|s_k)·s_k / Σ p(y|s_k)
# ──────────────────┼───────────────────────────────┼───────────────────────────────
# Calibration       │ Built into training            │ Self-calib (default) or
#                   │                                │ --calib_dir for external images
#
# Memory note (K=800):
#   Training allocates (B, K, P, P) UNet output per batch.
#   With B=32, K=800, P=64 → ~840 MB on GPU.
#   Reduce --batch_size if you hit OOM.  Default is 32 (8GB GPU).
#
# Requirements: torch>=2.0.0  tifffile  matplotlib  numpy
# Usage:
#   python test_sem.py                    # generate synthetic test image
#   python denoise_PN2V_juglab.py         # self-calibration (default)
#   python denoise_PN2V_juglab.py --calib_dir ./sem_images
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
        print(f"  Loaded calibration image: {os.path.basename(path)}  shape={img.shape}")
    return images


# ============================================================
# 2. 4-Neighbor Mean (signal proxy for histogram building)
# ============================================================

def _compute_4neighbor_mean(image: np.ndarray) -> np.ndarray:
    """
    Compute the 4-neighbor mean as a low-noise signal proxy.
    For each pixel (r,c): s_proxy = mean(up, down, left, right).
    Uses reflect padding so border pixels have valid neighbors.
    """
    img_t = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    kernel = torch.zeros(1, 1, 3, 3)
    kernel[0, 0, 0, 1] = 0.25   # up
    kernel[0, 0, 2, 1] = 0.25   # down
    kernel[0, 0, 1, 0] = 0.25   # left
    kernel[0, 0, 1, 2] = 0.25   # right
    with torch.no_grad():
        s_proxy = F.conv2d(img_t, kernel, padding=1)
    return s_proxy.squeeze().numpy()


# ============================================================
# 3. Histogram Noise Model  (juglab PN2V — non-parametric)
# ============================================================

class HistogramNoiseModel:
    """
    Non-parametric 2D conditional histogram noise model p(y | s).

    The histogram H[i, j] stores p(y_j | s_i), where rows index signal bins
    and columns index observation bins (both over [0, 1] with n_bins each).

    Building
    --------
    For each pixel pair (s_proxy, y_obs):
      - s_proxy = 4-neighbor mean of y_obs (signal estimate without center pixel)
      - y_obs   = original noisy pixel
    The 2D histogram counts how often signal s_i co-occurs with noise y_j.
    Each row is then L1-normalized to give a conditional probability.

    Likelihood query (differentiable)
    ---------------------------------
    Given predicted signal s (continuous, float) and observation y (observed),
    we look up p(y | s) using bilinear interpolation along the signal axis:

        s_float = s * (n_bins - 1)
        s_lo, s_hi = floor(s_float), ceil(s_float)
        frac = s_float - s_lo                        # ← gradient flows here
        p(y|s) ≈ H[s_lo, y_bin] * (1-frac) + H[s_hi, y_bin] * frac

    The gradient d(log p)/d(s) = (H[s_hi, y_bin] - H[s_lo, y_bin]) * (n_bins-1) / p
    is non-zero as long as adjacent rows differ, which guides the UNet to output
    signal values consistent with the observed noisy pixel.

    No parameters are trained — the histogram is built once and held fixed.
    """

    def __init__(self, n_bins: int = 256):
        self.n_bins    = n_bins
        self.histogram: Optional[np.ndarray] = None  # (n_bins, n_bins) float32
        self._hist_tensor: Optional[torch.Tensor] = None  # cached device tensor

    def build(self, images: List[np.ndarray]) -> None:
        """
        Build the 2D histogram from one or more normalized [0,1] images.
        Can be called with a single image (self-calibration) or a list of
        calibration images from --calib_dir.
        """
        n = self.n_bins
        hist = np.zeros((n, n), dtype=np.float64)

        for img in images:
            s_proxy = _compute_4neighbor_mean(img)
            y_flat  = img.flatten()
            s_flat  = s_proxy.flatten()

            s_bins = np.clip((s_flat * (n - 1)).astype(int), 0, n - 1)
            y_bins = np.clip((y_flat * (n - 1)).astype(int), 0, n - 1)
            np.add.at(hist, (s_bins, y_bins), 1)

        hist += 1e-30  # avoid empty rows collapsing to zero after normalization

        row_sums = hist.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-20)
        hist /= row_sums

        self.histogram = hist.astype(np.float32)
        self._hist_tensor = None  # invalidate cache

        n_pairs = sum(img.size for img in images)
        print(f"Histogram built: {n}×{n} bins  from {n_pairs:,} pixel pairs "
              f"across {len(images)} image(s)")

    def _ensure_tensor(self, device: torch.device) -> torch.Tensor:
        if self._hist_tensor is None or self._hist_tensor.device != device:
            assert self.histogram is not None, "Call build() before using the noise model"
            self._hist_tensor = torch.from_numpy(self.histogram).to(device)
        return self._hist_tensor

    def log_likelihood(self, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Compute log p(y | s) for each (y_i, s_i) pair.

        Args:
            y : observed noisy values, shape (N,), range [0, 1]
            s : predicted signal values, shape (N,), range [0, 1]
        Returns:
            log p(y | s), shape (N,)  [differentiable in s via bilinear interp]
        """
        hist   = self._ensure_tensor(y.device)
        n_bins = self.n_bins

        # Signal: continuous bin index (gradient flows here)
        s_float = s.clamp(0.0, 1.0) * (n_bins - 1)     # (N,), float
        s_lo    = s_float.long().clamp(0, n_bins - 2)   # (N,), integer, no grad
        s_hi    = (s_lo + 1).clamp(0, n_bins - 1)       # (N,), integer, no grad
        frac    = s_float - s_lo.float()                 # (N,), float, has grad

        # Observation: discrete bin index (no gradient needed through y)
        y_bin = (y.clamp(0.0, 1.0) * (n_bins - 1)).long().clamp(0, n_bins - 1)  # (N,)

        # Bilinear interpolation along signal axis
        p_lo = hist[s_lo, y_bin]                          # (N,), no grad (lookup)
        p_hi = hist[s_hi, y_bin]                          # (N,), no grad (lookup)
        p    = p_lo * (1.0 - frac) + p_hi * frac          # (N,), has grad via frac
        p    = p.clamp(min=1e-30)

        return torch.log(p)                               # (N,)

    @torch.no_grad()
    def plot(self, save_path: str = "data/histogram_noise_model.png") -> None:
        """Visualize the 2D histogram and the marginal noise std curve."""
        if self.histogram is None:
            print("Histogram not built yet — skipping plot.")
            return
        n = self.n_bins
        s_vals  = np.linspace(0, 1, n)
        y_vals  = np.linspace(0, 1, n)

        # Expected mean and std of p(y|s) for each signal bin
        means = (self.histogram * y_vals[None, :]).sum(axis=1)   # (n,)
        stds  = np.sqrt(
            np.maximum(
                (self.histogram * y_vals[None, :] ** 2).sum(axis=1) - means ** 2,
                0.0,
            )
        )

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        im = axes[0].imshow(
            np.log1p(self.histogram.T * 1000),
            origin='lower', aspect='auto',
            extent=[0, 1, 0, 1], cmap='hot',
        )
        axes[0].set_xlabel("Signal s")
        axes[0].set_ylabel("Observation y")
        axes[0].set_title("log(1 + 1000·H[s,y])  (2D histogram)")
        plt.colorbar(im, ax=axes[0])

        axes[1].plot(s_vals, stds, color='royalblue', label='noise std σ(s)')
        axes[1].plot(s_vals, means - s_vals, color='tomato', label='mean bias μ(s)−s')
        axes[1].axhline(0, color='k', lw=0.5)
        axes[1].set_xlabel("Signal s")
        axes[1].set_ylabel("Value")
        axes[1].set_title("Noise std and mean bias vs. signal")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Histogram plot saved: {save_path}")


# ============================================================
# 4. PN2VUNet — K output channels (K=800 prior samples)
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
    """
    4-level UNet encoder-decoder producing K output channels per pixel.

    Each output channel s_k represents one sample from the network's
    predictive prior distribution p(s | context), where context is the
    receptive field excluding the centre pixel (blind-spot masking ensures
    the centre pixel does not directly influence its own reconstruction).

    The K channels are NOT independent random draws — the network learns
    a deterministic mapping from context to K diverse signal hypotheses.
    Their diversity emerges from learning to maximise the negative log-evidence
    loss: -log(1/K Σ_k p(y|s_k)).  Clusters of identical samples would all
    receive the same likelihood weight and waste capacity.

    Input must be spatially divisible by 16 (4 MaxPool2d layers of stride 2).
    """

    def __init__(self, in_channels: int = 1, base_features: int = 32, K: int = 800):
        super().__init__()
        f       = base_features
        self.K  = K

        # Encoder
        self.enc1 = DoubleConvBlock(in_channels, f)
        self.enc2 = DoubleConvBlock(f,      f * 2)
        self.enc3 = DoubleConvBlock(f * 2,  f * 4)
        self.enc4 = DoubleConvBlock(f * 4,  f * 8)
        self.pool = nn.MaxPool2d(2)

        # Decoder
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

        # K-sample output head
        self.head = nn.Conv2d(f, K, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, 1, H, W)
        Returns:
            (B, K, H, W)  — K signal-sample predictions per pixel
        """
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)   # (B, K, H, W)


# ============================================================
# 5. N2V Dataset — blind-spot masking (same as other scripts)
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
# 6. Training — Negative Log-Evidence Loss
# ============================================================

def train_pn2v_juglab(
    model:         PN2VUNet,
    noise_model:   HistogramNoiseModel,
    image:         np.ndarray,
    patch_size:    int   = 64,
    batch_size:    int   = 32,
    num_epochs:    int   = 200,
    learning_rate: float = 3e-4,
    val_frac:      float = 0.1,
    device:        torch.device = None,
) -> PN2VUNet:
    """
    Train PN2VUNet with the negative log-evidence loss (Krull et al., 2020).

    Loss per masked pixel:
        L_i = -log( (1/K) Σ_{k=1}^{K} p_hist(y_i | s_i^k) )
            = -logsumexp_k( log p_hist(y_i | s_i^k) ) + log(K)

    where:
      y_i   = original noisy pixel (observed)
      s_i^k = k-th channel of UNet output at position i (signal sample)
      p_hist = histogram noise model likelihood (bilinear interpolation)

    Gradients flow through the bilinear interpolation of the histogram,
    guiding the network to predict s_k values where p_hist(y|s_k) is large.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    patches_per_epoch = 2000
    n_val   = max(1, int(patches_per_epoch * val_frac))
    n_train = patches_per_epoch - n_val

    train_ds = N2VDataset(image, patch_size=patch_size, num_patches=n_train, rng_seed=42)
    val_ds   = N2VDataset(image, patch_size=patch_size, num_patches=n_val,   rng_seed=99)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == 'cuda'),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0,
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6,
    )

    K = model.K
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Device:          {device}")
    print(f"UNet parameters: {n_params:,}  |  K (output samples): {K}")
    print(f"patch_size={patch_size}  batch_size={batch_size}  epochs={num_epochs}")
    print(f"Patches/epoch: train={n_train}  val={n_val}")
    print(f"Peak GPU tensor per batch: "
          f"~{batch_size * K * patch_size * patch_size * 4 / 1e6:.0f} MB  "
          f"(reduce --batch_size if OOM)")

    log_K = math.log(K)

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        # ── Training ──
        model.train()
        tr_loss, tr_count = 0.0, 0

        for noisy_in, noisy_tgt, mask in train_loader:
            noisy_in  = noisy_in.to(device)   # (B, 1, P, P)
            noisy_tgt = noisy_tgt.to(device)  # (B, 1, P, P)
            mask      = mask.to(device)        # (B, 1, P, P)

            optimizer.zero_grad()
            pred = model(noisy_in)             # (B, K, P, P)

            # Extract values at masked positions
            mask_2d  = mask.bool().squeeze(1)  # (B, P, P)
            y_obs    = noisy_tgt.squeeze(1)[mask_2d]     # (N_masked,)
            # (B, P, P, K) → index with mask_2d → (N_masked, K)
            s_samp   = pred.permute(0, 2, 3, 1)[mask_2d]  # (N_masked, K)

            # Compute log p(y | s_k) for every sample
            N_masked = y_obs.shape[0]
            y_exp    = y_obs.unsqueeze(1).expand(N_masked, K)   # (N_masked, K)
            log_liks = noise_model.log_likelihood(
                y_exp.reshape(-1), s_samp.reshape(-1),
            ).reshape(N_masked, K)                              # (N_masked, K)

            # Negative log-evidence: -logsumexp + log(K)
            log_ev = torch.logsumexp(log_liks, dim=1) - log_K  # (N_masked,)
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
    pad     = max(0, tile_size - image_size)
    padded  = image_size + pad
    remainder = padded % divisor
    if remainder != 0:
        pad += divisor - remainder
    return pad


@torch.no_grad()
def _mmse_from_samples(
    pred_K:      torch.Tensor,   # (K, H, W) float
    y_tile:      torch.Tensor,   # (H, W)    float
    noise_model: HistogramNoiseModel,
    device:      torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute MMSE posterior mean and prior mean from K network samples.

    MMSE(x) = Σ_k p(y_x | s_x^k) · s_x^k  /  Σ_k p(y_x | s_x^k)

    Returns:
        mmse_tile  : (H, W) numpy float32 — MMSE posterior estimate
        prior_tile : (H, W) numpy float32 — unweighted prior mean
    """
    K, H, W = pred_K.shape
    N       = H * W

    s_flat = pred_K.reshape(K, N)             # (K, N)
    y_flat = y_tile.reshape(N).unsqueeze(0).expand(K, N)  # (K, N)

    log_liks = noise_model.log_likelihood(
        y_flat.reshape(-1), s_flat.reshape(-1),
    ).reshape(K, N)                           # (K, N)

    weights   = torch.softmax(log_liks, dim=0)  # (K, N)
    mmse_flat = (weights * s_flat).sum(dim=0)   # (N,)
    prior_flat = s_flat.mean(dim=0)             # (N,)

    mmse  = mmse_flat.cpu().numpy().reshape(H, W).astype(np.float32)
    prior = prior_flat.cpu().numpy().reshape(H, W).astype(np.float32)
    return mmse, prior


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
    Tiled inference with Hann-window blending.

    Returns:
        mmse_image  : (H, W) — MMSE posterior mean (primary output)
        prior_image : (H, W) — prior mean (unweighted sample average, diagnostic)

    Note on infer_batch_size:
        Each tile produces a (B, K, H_t, W_t) tensor.  For K=800 and 256×256 tiles,
        B=1 uses ~209 MB; B=2 uses ~418 MB.  Default is 2 (conservative for 8GB GPU).
        Reduce to 1 if OOM; increase if you have spare VRAM.
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

            mmse_tile, prior_tile = _mmse_from_samples(
                preds[j], y_tile, noise_model, device,
            )
            mmse_sum[r:r + th, c:c + tw]  += mmse_tile.astype(np.float64) * hann_2d
            prior_sum[r:r + th, c:c + tw] += prior_tile.astype(np.float64) * hann_2d
            weight_sum[r:r + th, c:c + tw] += hann_2d

        done = min(i + infer_batch_size, total_tiles)
        if done % max(infer_batch_size, total_tiles // 5 or 1) == 0 \
                or done == total_tiles:
            print(f"  Inference: {done}/{total_tiles} tiles")

    denom     = np.maximum(weight_sum, 1e-8)
    mmse_img  = (mmse_sum  / denom).astype(np.float32)[:H, :W]
    prior_img = (prior_sum / denom).astype(np.float32)[:H, :W]
    return mmse_img, prior_img


# ============================================================
# 8. Save Outputs
# ============================================================

def save_outputs(
    image:      np.ndarray,
    mmse:       np.ndarray,
    prior:      np.ndarray,
    img_min:    float,
    img_max:    float,
    tif_path:   str = "data/denoised_sem_pn2v_juglab.tif",
    prior_path: str = "data/denoised_sem_pn2v_juglab_prior.tif",
    png_path:   str = "data/denoised_sem_pn2v_juglab_comparison.png",
) -> None:
    rng = img_max - img_min

    mmse_orig  = (mmse  * rng + img_min).astype(np.float32)
    prior_orig = (prior * rng + img_min).astype(np.float32)

    tifffile.imwrite(tif_path,   mmse_orig)
    tifffile.imwrite(prior_path, prior_orig)
    print(f"Saved MMSE output: {tif_path}   range: [{mmse_orig.min():.3f}, {mmse_orig.max():.3f}]")
    print(f"Saved prior mean:  {prior_path}  range: [{prior_orig.min():.3f}, {prior_orig.max():.3f}]")

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original SEM'); axes[0].axis('off')
    axes[1].imshow(prior, cmap='gray')
    axes[1].set_title('Prior mean (K samples avg)'); axes[1].axis('off')
    axes[2].imshow(mmse,  cmap='gray')
    axes[2].set_title('MMSE posterior (juglab PN2V)'); axes[2].axis('off')
    diff = np.abs(image - mmse) * 3
    axes[3].imshow(diff, cmap='hot')
    axes[3].set_title('|Original − MMSE| × 3'); axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Saved comparison:  {png_path}")


# ============================================================
# 9. Main Pipeline
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "PN2V juglab-faithful: non-parametric histogram noise model + "
            "K=800 sample UNet + MMSE posterior inference."
        )
    )
    parser.add_argument('--input',       type=str, default='data/test_sem.tif',
                        help='Path to input .tif/.tiff/.png image')
    parser.add_argument('--output',      type=str, default='',
                        help='Output .tif path (default: data/denoised_sem_pn2v_juglab.tif)')
    parser.add_argument('--calib_dir',   type=str, default='',
                        help='Directory of calibration images for building the histogram '
                             '(default: use input image itself — self-calibration)')
    parser.add_argument('--n_bins',      type=int, default=256,
                        help='Number of histogram bins per axis (default: 256)')
    parser.add_argument('--K',           type=int, default=800,
                        help='Number of UNet output samples (default: 800 as in official)')
    parser.add_argument('--epochs',      type=int, default=200)
    parser.add_argument('--patch_size',  type=int, default=64,
                        help='Training patch size; must be divisible by 16')
    parser.add_argument('--batch_size',  type=int, default=32,
                        help='Training batch size (reduce if OOM; default: 32 for 8GB GPU)')
    parser.add_argument('--tile_size',   type=int, default=256,
                        help='Inference tile size applied to both H and W')
    parser.add_argument('--tile_overlap',type=int, default=32)
    parser.add_argument('--infer_batch', type=int, default=2,
                        help='Inference batch size (reduce to 1 if GPU OOM; default: 2)')
    parser.add_argument('--device',      type=str, default=None,
                        help='Device override: cuda, cpu, cuda:1 … (default: auto)')
    args = parser.parse_args()

    input_path   = args.input
    output_path  = args.output or 'data/denoised_sem_pn2v_juglab.tif'
    prior_path   = output_path.replace('.tif', '_prior.tif')
    png_path     = output_path.replace('.tif', '_comparison.png')

    os.makedirs('data', exist_ok=True)
    device = torch.device(
        args.device if args.device
        else ('cuda' if torch.cuda.is_available() else 'cpu')
    )

    # ── Load input image ──
    image, img_min, img_max = load_sem_image(input_path)
    print(f"Input image: {image.shape}  range: [{img_min:.3f}, {img_max:.3f}]")

    # ── Build histogram noise model ──
    noise_model = HistogramNoiseModel(n_bins=args.n_bins)
    if args.calib_dir:
        print(f"Building histogram from calibration directory: {args.calib_dir}")
        calib_images = load_images_from_dir(args.calib_dir)
    else:
        print("Building histogram from input image (self-calibration)")
        calib_images = [image]
    noise_model.build(calib_images)
    noise_model._ensure_tensor(device)  # pre-load histogram to GPU

    # ── Build model ──
    model = PN2VUNet(in_channels=1, base_features=32, K=args.K)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"PN2VUNet: K={args.K}  params={n_params:,}")

    # ── Validate patch size ──
    patch_size = args.patch_size
    if patch_size % 16 != 0:
        patch_size = (patch_size // 16 + 1) * 16
        print(f"patch_size rounded up to {patch_size} (must be divisible by 16)")

    # ── Train ──
    train_pn2v_juglab(
        model, noise_model, image,
        patch_size=patch_size,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        device=device,
    )

    # ── Inference ──
    print("\nRunning tiled MMSE inference ...")
    mmse_img, prior_img = predict_tiled(
        model, image, noise_model,
        tile_size=(args.tile_size, args.tile_size),
        tile_overlap=(args.tile_overlap, args.tile_overlap),
        infer_batch_size=args.infer_batch,
        device=device,
    )

    # ── Save ──
    save_outputs(
        image, mmse_img, prior_img, img_min, img_max,
        tif_path=output_path,
        prior_path=prior_path,
        png_path=png_path,
    )
    noise_model.plot(save_path='data/histogram_noise_model.png')


if __name__ == '__main__':
    main()

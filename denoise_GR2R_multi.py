# ============================================================
# SEM Image Denoising — GR2R, Multi-Image (pure PyTorch)
# ============================================================
# Based on:
#   "Recorrupted-to-Recorrupted: Unsupervised Deep Learning for Image Denoising"
#   Guo et al., CVPR 2021  (arXiv:2102.02234)
#
# Train ONCE on a batch of images acquired under similar conditions,
# then denoise every image with the same model.
#
# Differences from denoise_GR2R.py:
#   + MultiImageGR2RDataset: patches drawn from a pool of images, each pair
#     independently re-corrupted (no blind-spot masking)
#   + Noise std auto-estimated per image; pooled mean used as shared sigma_r
#     (override with --noise_std)
#   + --save_model / --load_model checkpoint support
#   + --train_dir allows training on a subset, inference on all
#   + --input_dir / --output_dir replace --input / --output
#   + Per-image output ({stem}_denoised.tif, {stem}_comparison.png)
#
# Identical to denoise_GR2R.py:
#   = estimate_noise_std, recorrupt_gaussian, recorrupt_poisson
#   = DoubleConvBlock, N2VUNet architecture (4-level, base_features=32)
#   = Batched tiled inference with per-axis reflection padding (Opt 4)
#   = Hann-window blending (Opt 2)
#   = MSE over ALL pixels (no masking needed in GR2R)
#   = Adam + CosineAnnealingLR, lr=4e-4 → 1e-6
#
# Requirements: torch>=2.0.0  tifffile  matplotlib  numpy
#
# Usage:
#   python denoise_GR2R_multi.py --input_dir ./sem_images --output_dir ./denoised
#   python denoise_GR2R_multi.py --input_dir ./sem_images --output_dir ./denoised \
#                                --epochs 100 --patch_size 64 --poisson
#
#   # Train on 2 representative images, save model, denoise all 10 later:
#   python denoise_GR2R_multi.py --input_dir ./train_imgs --output_dir ./denoised \
#                                --save_model sem_gr2r.pt
#   python denoise_GR2R_multi.py --input_dir ./all_imgs   --output_dir ./denoised \
#                                --load_model sem_gr2r.pt
# ============================================================

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import argparse
import time
from pathlib import Path
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


def find_images(directory: str) -> List[Path]:
    """Return all .tif/.tiff/.png files in directory, sorted by name."""
    d = Path(directory)
    images = sorted(
        p for p in d.iterdir()
        if p.suffix.lower() in {'.tif', '.tiff', '.png'}
    )
    if not images:
        raise FileNotFoundError(f"No .tif/.tiff/.png images found in: {directory}")
    return images


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
    """4-level encoder-decoder UNet for GR2R (2D grayscale)."""

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


def recorrupt_gaussian(
    patch: np.ndarray,
    sigma_r: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add i.i.d. Gaussian noise N(0, sigma_r²) and clip to [0, 1]."""
    noise = rng.standard_normal(patch.shape).astype(np.float32) * sigma_r
    return np.clip(patch + noise, 0.0, 1.0)


def recorrupt_poisson(
    patch: np.ndarray,
    photon_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Resample each pixel from Poisson(y * photon_scale) / photon_scale.
    Matching SEM shot-noise physics: re-corruption has same mean and signal-proportional variance.
    """
    counts    = np.maximum(patch * photon_scale, 0.0)
    resampled = rng.poisson(counts).astype(np.float32) / photon_scale
    return np.clip(resampled, 0.0, 1.0)


# ============================================================
# 4. Multi-Image GR2R Dataset
# ============================================================

class MultiImageGR2RDataset(Dataset):
    """
    Self-supervised GR2R dataset over a pool of images.

    Each item is a (y1, y2) pair where y1 and y2 are two independently
    re-corrupted versions of the same patch drawn from a randomly selected
    image in the pool.

    Key difference from MultiImageN2VDataset: NO blind-spot masking.
    The network receives the full patch and predicts an independent re-corruption.
    MSE is computed over ALL pixels — the full-context receptive field is exploited.

    Images too small for patch_size are skipped with a warning.
    """

    def __init__(
        self,
        images: List[np.ndarray],
        patch_size:   int   = 64,
        num_patches:  int   = 2000,
        sigma_r:      float = 0.05,
        photon_scale: float = 100.0,
        use_poisson:  bool  = False,
        rng_seed:     int   = None,
    ):
        assert patch_size % 8 == 0, f"patch_size must be divisible by 8, got {patch_size}"

        self.images = []
        for i, img in enumerate(images):
            if img.shape[0] >= patch_size and img.shape[1] >= patch_size:
                self.images.append(img)
            else:
                print(f"  [WARNING] Image #{i} shape {img.shape} is smaller than "
                      f"patch_size={patch_size} — skipped for training.")

        if not self.images:
            raise ValueError(
                f"All images are smaller than patch_size={patch_size}. "
                "Reduce patch_size or use larger images."
            )

        self.patch_size   = patch_size
        self.num_patches  = num_patches
        self.sigma_r      = sigma_r
        self.photon_scale = photon_scale
        self.use_poisson  = use_poisson
        self.rng          = np.random.default_rng(rng_seed)
        self.shapes       = [(img.shape[0], img.shape[1]) for img in self.images]
        self.n_images     = len(self.images)

    def __len__(self) -> int:
        return self.num_patches

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        P = self.patch_size

        img_idx = int(self.rng.integers(0, self.n_images))
        H, W    = self.shapes[img_idx]
        r0      = int(self.rng.integers(0, H - P))
        c0      = int(self.rng.integers(0, W - P))
        patch   = self.images[img_idx][r0:r0 + P, c0:c0 + P].copy()

        if self.use_poisson:
            y1 = recorrupt_poisson(patch, self.photon_scale, self.rng)
            y2 = recorrupt_poisson(patch, self.photon_scale, self.rng)
        else:
            y1 = recorrupt_gaussian(patch, self.sigma_r, self.rng)
            y2 = recorrupt_gaussian(patch, self.sigma_r, self.rng)

        return (
            torch.from_numpy(y1).unsqueeze(0),   # (1, P, P) — network input
            torch.from_numpy(y2).unsqueeze(0),   # (1, P, P) — training target
        )


# ============================================================
# 5. Training Loop
# ============================================================

def train_gr2r_multi(
    model: nn.Module,
    images: List[np.ndarray],
    patch_size:     int   = 64,
    batch_size:     int   = 128,
    num_epochs:     int   = 100,
    learning_rate:  float = 4e-4,
    sigma_r:        float = 0.05,
    photon_scale:   float = 100.0,
    use_poisson:    bool  = False,
    val_percentage: float = 0.1,
    device: torch.device  = None,
) -> nn.Module:
    """Train GR2R jointly on multiple images (one training run for all)."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    patches_per_epoch = max(2000, 500 * len(images))
    n_val   = max(1, int(patches_per_epoch * val_percentage))
    n_train = patches_per_epoch - n_val

    common_kw = dict(
        patch_size=patch_size,
        sigma_r=sigma_r,
        photon_scale=photon_scale,
        use_poisson=use_poisson,
    )
    train_ds = MultiImageGR2RDataset(images, num_patches=n_train, rng_seed=42, **common_kw)
    val_ds   = MultiImageGR2RDataset(images, num_patches=n_val,   rng_seed=99, **common_kw)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    loss_fn   = nn.MSELoss()   # mean over all pixels (no masking needed in GR2R)

    mode_str = f"Poisson(scale={photon_scale:.0f})" if use_poisson \
               else f"Gaussian(σ_r={sigma_r:.5f})"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nDevice: {device}  |  Model parameters: {n_params:,}")
    print(f"Training on {len(train_ds.images)} image(s)")
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
# 6. Tiled Inference
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
    axes[0].imshow(image,    cmap='gray'); axes[0].set_title('Original');          axes[0].axis('off')
    axes[1].imshow(denoised, cmap='gray'); axes[1].set_title('GR2R Denoised');     axes[1].axis('off')
    diff = np.abs(image - denoised) * 3
    axes[2].imshow(diff,     cmap='hot');  axes[2].set_title('Difference (×3)');   axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  Saved PNG: {png_path}")


# ============================================================
# 8. Main Pipeline
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="GR2R multi-image SEM denoiser: train once, denoise all."
    )
    parser.add_argument('--input_dir',    type=str, default='.',
                        help='Directory with input .tif/.tiff/.png images (used for both '
                             'training and inference unless --train_dir is specified)')
    parser.add_argument('--train_dir',    type=str, default='',
                        help='Optional: separate directory of images used ONLY for training. '
                             'All images in --input_dir will still be denoised.')
    parser.add_argument('--output_dir',   type=str, default='denoised',
                        help='Directory to write denoised results')
    parser.add_argument('--patch_size',   type=int,   default=64)
    parser.add_argument('--batch_size',   type=int,   default=128)
    parser.add_argument('--epochs',       type=int,   default=100)
    parser.add_argument('--tile_size',    type=int,   default=256,
                        help='Inference tile size (applied to both H and W)')
    parser.add_argument('--tile_overlap', type=int,   default=48)
    parser.add_argument('--alpha',        type=float, default=1.0,
                        help='Re-corruption strength: sigma_r = alpha × mean(per-image sigma). '
                             'Lower (0.5) is gentler; higher (2.0) adds more supervision signal. '
                             'Default 1.0 works well in practice.')
    parser.add_argument('--noise_std',    type=float, default=0.0,
                        help='Manual noise std override (0 = auto-estimate per image and average). '
                             'Set this if the noise level is known precisely.')
    parser.add_argument('--poisson',      action='store_true',
                        help='Use Poisson re-corruption instead of Gaussian. '
                             'Recommended when SEM noise is dominated by shot noise.')
    parser.add_argument('--photon_scale', type=float, default=100.0,
                        help='Photon scale for Poisson re-corruption (used only with --poisson). '
                             'Higher values → less variance per re-corruption.')
    parser.add_argument('--save_model',   type=str, default='',
                        help='Path to save trained model weights (.pt)')
    parser.add_argument('--load_model',   type=str, default='',
                        help='Path to load pre-trained weights — skips training entirely')
    parser.add_argument('--device',       type=str, default=None,
                        help='Device override: cuda, cpu, cuda:1 … (default: auto)')
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    )

    # ── 1. Discover inference images ─────────────────────────────────────────
    infer_paths = find_images(args.input_dir)
    print(f"Images to denoise ({len(infer_paths)}) from '{args.input_dir}':")
    for p in infer_paths:
        print(f"  {p.name}")

    # ── 2. Discover training images ──────────────────────────────────────────
    if args.train_dir:
        train_paths = find_images(args.train_dir)
        print(f"\nTraining images ({len(train_paths)}) from '{args.train_dir}':")
        for p in train_paths:
            print(f"  {p.name}")
    else:
        train_paths = infer_paths

    # ── 3. Load training images & estimate noise ──────────────────────────────
    print("\nLoading training images...")
    train_images  = []
    train_meta    = []   # (img_min, img_max) per training image
    sigma_per_img = []
    for p in train_paths:
        img, img_min, img_max = load_sem_image(str(p))
        sigma_i = estimate_noise_std(img)
        train_images.append(img)
        train_meta.append((img_min, img_max))
        sigma_per_img.append(sigma_i)
        print(f"  {p.name}: shape={img.shape}  "
              f"range=[{img_min:.1f},{img_max:.1f}]  σ_est={sigma_i:.5f}")

    # Determine shared sigma_r
    if args.noise_std > 0:
        sigma_shared = args.noise_std
        print(f"\nNoise std (manual):    σ = {sigma_shared:.5f}")
    else:
        sigma_shared = float(np.mean(sigma_per_img))
        print(f"\nNoise std (mean of {len(sigma_per_img)} estimates): σ = {sigma_shared:.5f}")

    sigma_r = args.alpha * sigma_shared
    print(f"Re-corruption std:     σ_r = α × σ = {args.alpha:.2f} × {sigma_shared:.5f} = {sigma_r:.5f}")

    if args.poisson:
        print(f"Re-corruption mode: Poisson (photon_scale={args.photon_scale:.0f})")
    else:
        print(f"Re-corruption mode: Gaussian")

    # ── 4. Build model ────────────────────────────────────────────────────────
    model = N2VUNet(in_channels=1, base_features=32)

    if args.load_model and os.path.isfile(args.load_model):
        model.load_state_dict(torch.load(args.load_model, map_location=device))
        model = model.to(device)
        print(f"\nLoaded pre-trained weights: {args.load_model}  (skipping training)")
    else:
        # ── 5. Train ONCE on all training images ──────────────────────────────
        model = train_gr2r_multi(
            model, train_images,
            patch_size=args.patch_size,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            sigma_r=sigma_r,
            photon_scale=args.photon_scale,
            use_poisson=args.poisson,
            device=device,
        )
        if args.save_model:
            torch.save(model.state_dict(), args.save_model)
            print(f"Model weights saved: {args.save_model}")

    # ── 6. Load inference images (if different from training set) ─────────────
    if args.train_dir:
        print("\nLoading inference images...")
        infer_images = []
        infer_meta   = []
        for p in infer_paths:
            img, img_min, img_max = load_sem_image(str(p))
            infer_images.append(img)
            infer_meta.append((img_min, img_max))
            print(f"  {p.name}: shape={img.shape}")
    else:
        infer_images = train_images
        infer_meta   = train_meta

    # ── 7. Inference on every image ───────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tile_size    = (args.tile_size, args.tile_size)
    tile_overlap = (args.tile_overlap, args.tile_overlap)

    print(f"\nRunning inference on {len(infer_paths)} image(s)...")
    for i, (p, img, (img_min, img_max)) in enumerate(
            zip(infer_paths, infer_images, infer_meta)):
        print(f"\n[{i+1}/{len(infer_paths)}] {p.name}")
        denoised = predict_tiled(
            model, img,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            device=device,
        )
        tif_path = str(out_dir / f"{p.stem}_denoised.tif")
        png_path = str(out_dir / f"{p.stem}_comparison.png")
        save_outputs(img, denoised, img_min, img_max, tif_path, png_path)

    print(f"\nDone. All results saved to '{out_dir}/'")


if __name__ == '__main__':
    main()

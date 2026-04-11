# ============================================================
# SEM Image Denoising — Noise2Void v2 (pure PyTorch)
# Optimizations over denoise_N2V.py:
#   1. Vectorized blind-spot masking  (CPU bottleneck removed)
#   2. Batched tiled inference        (GPU utilization improved)
#   4. Reflection padding in predict  (small/odd-size images handled)
# ============================================================
# Requirements: torch>=2.0.0  tifffile  matplotlib  numpy
# Usage:
#   python test_sem.py       # generate synthetic test image
#   python denoise_N2V_v2.py # train + denoise -> denoised_sem_N2V_v2.tif
# ============================================================

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
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

torch.set_float32_matmul_precision('high')  # 啟用 Tensor Core，提升 RTX 訓練速度


# ============================================================
# 1. Image Loading
# ============================================================

def load_sem_image(path: str) -> Tuple[np.ndarray, float, float]:
    """Load SEM image, normalize to float32 [0, 1] grayscale numpy array.
    Also returns original min/max for restoring pixel values after denoising."""
    img = tifffile.imread(path).astype(np.float32)

    if img.ndim == 3 and img.shape[-1] == 3:
        img = img @ np.array([0.2989, 0.5870, 0.1140])

    img_min, img_max = float(img.min()), float(img.max())
    img = (img - img_min) / (img_max - img_min + 1e-8)
    return img, img_min, img_max


# ============================================================
# 2. UNet Architecture  (unchanged)
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
    """
    4-level encoder-decoder UNet for Noise2Void (2D grayscale).

    Encoder channel widths : 1 -> 32 -> 64 -> 128 -> 256 (bottleneck)
    Decoder                : skip-concat + DoubleConvBlock at each level
    Downsampling           : MaxPool2d(2)
    Upsampling             : bilinear Upsample(x2) + Conv2d(1x1) to halve channels

    Input spatial dimensions must be divisible by 8.
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
# 3. N2V Dataset — Optimization 1: Vectorized Masking
# ============================================================

class N2VDataset(Dataset):
    """
    Extracts random patches from a single image with N2V blind-spot masking.

    Optimization 1 vs denoise_N2V.py:
        _apply_n2v_masking now uses fully vectorized NumPy operations instead
        of a Python for/while loop, eliminating the CPU bottleneck that caused
        low GPU utilization during training.

        The (0,0) self-replacement guard is handled by randomly nudging either
        dr or dc (chosen per-pixel) so the corrected offset distribution is
        symmetric in both axes rather than always biasing toward vertical.
    """

    def __init__(
        self,
        image: np.ndarray,
        patch_size: int   = 64,
        num_patches: int  = 2000,
        mask_ratio: float = 0.006,
        neighbor_radius: int = 5,
        rng_seed: int = None,
    ):
        assert patch_size % 8 == 0, f"patch_size must be divisible by 8, got {patch_size}"
        assert image.shape[0] >= patch_size and image.shape[1] >= patch_size, \
            f"Image {image.shape} too small for patch_size={patch_size}"

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
        P = self.patch_size

        r0 = self.rng.integers(0, self.H - P)
        c0 = self.rng.integers(0, self.W - P)
        patch = self.image[r0:r0 + P, c0:c0 + P].copy()

        corrupted, mask = self._apply_n2v_masking(patch)

        return (
            torch.from_numpy(corrupted).unsqueeze(0),
            torch.from_numpy(patch).unsqueeze(0),
            torch.from_numpy(mask).unsqueeze(0),
        )

    def _apply_n2v_masking(
        self, patch: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized N2V blind-spot masking.

        All n_masked pixels are processed simultaneously via NumPy array ops.
        (0,0) self-replacement guard: when an offset lands on (0,0), we randomly
        nudge dr OR dc by ±1 (chosen per collision) to keep the offset
        distribution symmetric rather than always biasing one axis.
        """
        P   = self.patch_size
        rad = self.neighbor_radius

        corrupted = patch.copy()
        mask      = np.zeros((P, P), dtype=np.float32)

        flat_idx  = self.rng.choice(P * P, size=self.n_masked, replace=False)
        rows, cols = np.unravel_index(flat_idx, (P, P))

        # --- Vectorized offset sampling ---
        dr = self.rng.integers(-rad, rad + 1, size=self.n_masked)
        dc = self.rng.integers(-rad, rad + 1, size=self.n_masked)

        # Guard: fix any (dr, dc) == (0, 0) to avoid self-replacement
        zero_mask = (dr == 0) & (dc == 0)
        if np.any(zero_mask):
            n_fix = int(np.sum(zero_mask))
            # Randomly choose per-collision whether to shift dr or dc
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
# 4. Training Loop  (unchanged)
# ============================================================

def train_n2v(
    model: nn.Module,
    image: np.ndarray,
    patch_size:     int   = 64,
    batch_size:     int   = 128,
    num_epochs:     int   = 100,
    learning_rate:  float = 4e-4,
    val_percentage: float = 0.1,
    device: torch.device  = None,
) -> nn.Module:
    """Self-supervised N2V training on a single image."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    patches_per_epoch = 2000
    n_val   = max(1, int(patches_per_epoch * val_percentage))
    n_train = patches_per_epoch - n_val

    train_ds = N2VDataset(image, patch_size=patch_size, num_patches=n_train, rng_seed=42)
    val_ds   = N2VDataset(image, patch_size=patch_size, num_patches=n_val,   rng_seed=99)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    loss_fn   = nn.MSELoss(reduction='sum')

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}  |  Model parameters: {n_params:,}")
    print(f"Training: patch_size={patch_size}  batch_size={batch_size}  epochs={num_epochs}")
    print(f"Patches/epoch: train={n_train}  val={n_val}")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        model.train()
        tr_loss, tr_count = 0.0, 0
        for noisy_in, clean_tgt, mask in train_loader:
            noisy_in  = noisy_in.to(device)
            clean_tgt = clean_tgt.to(device)
            mask      = mask.to(device)

            optimizer.zero_grad()
            pred = model(noisy_in)
            loss = loss_fn(pred * mask, clean_tgt * mask)
            loss.backward()
            optimizer.step()

            tr_loss  += loss.item()
            tr_count += mask.sum().item()

        model.eval()
        vl_loss, vl_count = 0.0, 0
        with torch.no_grad():
            for noisy_in, clean_tgt, mask in val_loader:
                noisy_in  = noisy_in.to(device)
                clean_tgt = clean_tgt.to(device)
                mask      = mask.to(device)
                pred      = model(noisy_in)
                vl_loss  += loss_fn(pred * mask, clean_tgt * mask).item()
                vl_count += mask.sum().item()

        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            tr_mse = tr_loss / max(tr_count, 1)
            vl_mse = vl_loss / max(vl_count, 1)
            print(f"Epoch [{epoch:3d}/{num_epochs}]  "
                  f"train_loss={tr_mse:.6f}  val_loss={vl_mse:.6f}  "
                  f"elapsed={time.time() - t0:.1f}s")

    print("Training complete.")
    return model


# ============================================================
# 5. Tiled Inference — Optimization 2 (Batched) + 4 (Padding)
# ============================================================

def _compute_padding(image_size: int, tile_size: int) -> int:
    """
    Return the amount of reflection padding needed on one axis.

    Two concerns, handled independently:
      a) If image < tile, pad up to tile_size so at least one tile fits.
      b) Always ensure the padded dimension is divisible by 8 (UNet requirement
         when the whole padded axis is processed in a single tile pass).

    Note: concerns (a) and (b) are computed on the already-padded size to avoid
    the off-by-one bug where only one axis triggers (b).
    """
    pad = max(0, tile_size - image_size)
    padded = image_size + pad
    remainder = padded % 8
    if remainder != 0:
        pad += 8 - remainder
    return pad


def predict_tiled(
    model: nn.Module,
    image: np.ndarray,
    tile_size:       Tuple[int, int] = (256, 256),
    tile_overlap:    Tuple[int, int] = (48, 48),
    infer_batch_size: int            = 8,
    device: torch.device             = None,
) -> np.ndarray:
    """
    Run inference on large images by processing overlapping tiles.
    Overlapping regions are blended using a 2D Hann window to avoid seams.

    Optimization 4 — Reflection Padding:
        Images smaller than tile_size (or with dimensions not divisible by 8)
        are padded with reflect mode before inference and cropped afterward.
        Padding is computed per-axis independently so that the multiple-of-8
        alignment is always applied regardless of which axis needed tile padding.

    Optimization 2 — Batched GPU Inference:
        All tile coordinates are collected upfront. Tiles are stacked into
        batches of `infer_batch_size` and sent to the GPU in one forward pass,
        maximising CUDA occupancy and minimising CPU-GPU transfer overhead.
        The Hann-window blending is applied on CPU after each batch completes.

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

    # --- Optimization 4: Reflection padding (per-axis, independent) ---
    pad_h = _compute_padding(H, th)
    pad_w = _compute_padding(W, tw)

    if pad_h > 0 or pad_w > 0:
        padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
    else:
        padded = image

    pH, pW = padded.shape
    stride_h = th - oh
    stride_w = tw - ow

    # 2D Hann window for weighted blending
    hann_h  = torch.hann_window(th, periodic=False).numpy()
    hann_w  = torch.hann_window(tw, periodic=False).numpy()
    hann_2d = np.outer(hann_h, hann_w)                    # (th, tw)

    output_sum = np.zeros((pH, pW), dtype=np.float64)
    weight_sum = np.zeros((pH, pW), dtype=np.float64)

    # Build tile origin lists, ensuring the last tile reaches the padded edge
    row_starts: List[int] = list(range(0, pH - th + 1, stride_h))
    col_starts: List[int] = list(range(0, pW - tw + 1, stride_w))
    if row_starts[-1] + th < pH:
        row_starts.append(pH - th)
    if col_starts[-1] + tw < pW:
        col_starts.append(pW - tw)

    coords = [(r0, c0) for r0 in row_starts for c0 in col_starts]
    total_tiles = len(coords)
    processed   = 0

    # --- Optimization 2: Batched GPU inference ---
    model.eval()
    with torch.no_grad():
        for i in range(0, total_tiles, infer_batch_size):
            batch_coords = coords[i:i + infer_batch_size]

            # Stack tiles -> (B, 1, th, tw)
            tiles   = [padded[r0:r0 + th, c0:c0 + tw] for (r0, c0) in batch_coords]
            batch_t = torch.from_numpy(np.stack(tiles)).unsqueeze(1).to(device)

            # Single GPU forward pass for the whole batch
            preds = model(batch_t).squeeze(1).cpu().numpy()  # (B, th, tw)

            # Accumulate each tile's prediction with Hann-window weight
            for j, (r0, c0) in enumerate(batch_coords):
                output_sum[r0:r0 + th, c0:c0 + tw] += preds[j].astype(np.float64) * hann_2d
                weight_sum[r0:r0 + th, c0:c0 + tw] += hann_2d

            processed = min(i + infer_batch_size, total_tiles)
            if processed % max(infer_batch_size, total_tiles // 5 or 1) == 0 \
                    or processed == total_tiles:
                print(f"  Inference: {processed}/{total_tiles} tiles")

    denoised_pad = (output_sum / np.maximum(weight_sum, 1e-8)).astype(np.float32)

    # Crop back to original image dimensions (remove reflection padding)
    denoised = denoised_pad[:H, :W]
    return denoised


# ============================================================
# 6. Save Outputs  (unchanged)
# ============================================================

def save_outputs(
    image:    np.ndarray,
    denoised: np.ndarray,
    img_min:  float,
    img_max:  float,
    tif_path: str = "data/denoised_sem_N2V_v2.tif",
    png_path: str = "data/denoising_result_v2.png",
) -> None:
    """Save denoised TIF (original value range) and side-by-side comparison PNG."""
    denoised_original = (denoised * (img_max - img_min) + img_min).astype(np.float32)
    tifffile.imwrite(tif_path, denoised_original)
    print(f"Saved: {tif_path},  range: [{denoised_original.min():.3f}, {denoised_original.max():.3f}]")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image,    cmap='gray')
    axes[0].set_title('Original SEM Image')
    axes[0].axis('off')

    axes[1].imshow(denoised, cmap='gray')
    axes[1].set_title('N2V Denoised (v2)')
    axes[1].axis('off')

    diff = np.abs(image - denoised) * 3
    axes[2].imshow(diff, cmap='hot')
    axes[2].set_title('Difference (×3)')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.show()
    print(f"Saved: {png_path}")


# ============================================================
# 7. Main Pipeline
# ============================================================

def main(
    patch_size:       int             = 64,
    batch_size:       int             = 128,
    num_epochs:       int             = 100,
    tile_size:        Tuple[int, int] = (256, 256),
    tile_overlap:     Tuple[int, int] = (48, 48),
    infer_batch_size: int             = 8,
) -> None:
    input_path  = "data/test_sem.tif"
    output_path = "data/denoised_sem_N2V_v2.tif"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image, img_min, img_max = load_sem_image(input_path)
    print(f"Image shape: {image.shape},  range: [{img_min:.3f}, {img_max:.3f}]")

    model = N2VUNet(in_channels=1, base_features=32)

    model = train_n2v(
        model, image,
        patch_size=patch_size,
        batch_size=batch_size,
        num_epochs=num_epochs,
        device=device,
    )

    print("\nRunning tiled inference...")
    denoised = predict_tiled(
        model, image,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        infer_batch_size=infer_batch_size,
        device=device,
    )

    save_outputs(image, denoised, img_min, img_max, tif_path=output_path)


if __name__ == '__main__':
    main()

# ============================================================
# SEM Image Denoising — AP-BSN, Multi-Image (pure PyTorch)
# ============================================================
# Paper: "AP-BSN: Self-Supervised Denoising for Real-World Images
#         via Asymmetric PD and Blind-Spot Network"
#         Lee et al., CVPR 2022
#
# Extension: train ONCE on multiple images from the same instrument,
# then denoise all images with the shared model.
#
# Key difference from denoise_apbsn.py:
#   APBSNDataset (single PD image) →
#   MultiImageAPBSNDataset (pool of PD images, one per input image)
#
# Usage:
#   python denoise_apbsn_multi.py --input_dir ./sem_images --output_dir ./denoised
#
#   # Train on 2 images, save model, denoise 10 later:
#   python denoise_apbsn_multi.py --train_dir ./train_imgs --input_dir ./all_imgs \
#                                  --output_dir ./denoised --save_model apbsn.pt
#   python denoise_apbsn_multi.py --input_dir ./new_imgs   --output_dir ./denoised \
#                                  --load_model apbsn.pt
#
# pd_stride=2  → SEM (pixel-independent Poisson/Gaussian noise)
# pd_stride=5  → Real camera sRGB (spatially-correlated ISP noise)
# ============================================================
# Requirements: torch>=2.0  tifffile  matplotlib  numpy
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
import torch.nn.functional as F
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
# 2. Pixel-Shuffle Downsampling (PD) Operations
# ============================================================

def pd_downsample(
    x: torch.Tensor,
    stride: int,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Pixel-Shuffle Downsampling: (B, 1, H, W) → (B, r², H//r, W//r).
    Returns (pd_tensor, (pad_h, pad_w)).
    """
    B, C, H, W = x.shape
    r = stride
    pad_h = (r - H % r) % r
    pad_w = (r - W % r) % r
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    _, _, Hp, Wp = x.shape
    x = x.view(B, C, Hp // r, r, Wp // r, r)
    x = x.permute(0, 1, 3, 5, 2, 4).reshape(B, C * r * r, Hp // r, Wp // r)
    return x, (pad_h, pad_w)


def pd_upsample(
    x: torch.Tensor,
    stride: int,
    pad_hw: Tuple[int, int],
    orig_hw: Tuple[int, int],
) -> torch.Tensor:
    """
    Inverse Pixel-Shuffle Upsampling: (B, r², H//r, W//r) → (B, 1, H, W).
    Crops out the reflect-padding added by pd_downsample.
    """
    B, _Cr2, Hd, Wd = x.shape
    r = stride
    x = x.reshape(B, 1, r, r, Hd, Wd)
    x = x.permute(0, 1, 4, 2, 5, 3).reshape(B, 1, Hd * r, Wd * r)
    return x[:, :, :orig_hw[0], :orig_hw[1]]


def _numpy_pd(image: np.ndarray, r: int) -> np.ndarray:
    """
    PD transform on a 2-D numpy array (CPU, no GPU round-trip).
    result shape: (r², Hd, Wd)
    """
    H, W = image.shape
    pad_h = (r - H % r) % r
    pad_w = (r - W % r) % r
    if pad_h > 0 or pad_w > 0:
        image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
    Hp, Wp = image.shape
    Hd, Wd = Hp // r, Wp // r
    pd = image.reshape(Hd, r, Wd, r).transpose(1, 3, 0, 2).reshape(r * r, Hd, Wd)
    return pd.astype(np.float32)


# ============================================================
# 3. BSN U-Net Architecture  (identical to denoise_apbsn.py)
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


class BSNUNet(nn.Module):
    """4-level encoder-decoder U-Net for AP-BSN in the PD domain."""

    def __init__(self, pd_stride: int = 2, base_features: int = 32):
        super().__init__()
        nc = pd_stride * pd_stride
        f  = base_features

        self.enc1 = DoubleConvBlock(nc,    f)
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

        self.head = nn.Conv2d(f, nc, kernel_size=1)

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
# 4. Multi-Image AP-BSN Dataset
# ============================================================

class MultiImageAPBSNDataset(Dataset):
    """
    Self-supervised AP-BSN dataset pooling PD-domain images from multiple inputs.

    Each image is PD-transformed once in __init__ and stored as a (r², Hd, Wd)
    array. During training, patches are drawn uniformly at random across all
    stored PD images.

    Masking: for each selected spatial position (row, col), ALL r² channels are
    simultaneously replaced with values from a randomly chosen neighbour —
    preventing the BSN from exploiting cross-channel identity at that location.
    The binary mask shape is (1, P, P) and broadcasts over (B, r², P, P).

    Parameters
    ----------
    pd_images      : list of (r², Hd_i, Wd_i) float32 arrays
    patch_size     : must be divisible by 8 and ≤ min(Hd, Wd) across all images
    num_patches    : virtual epoch length
    mask_ratio     : fraction of spatial positions masked per patch
    neighbor_radius: max displacement for blind-spot replacement neighbour
    rng_seed       : optional seed for reproducibility
    """

    def __init__(
        self,
        pd_images: List[np.ndarray],
        patch_size: int = 64,
        num_patches: int = 2000,
        mask_ratio: float = 0.006,
        neighbor_radius: int = 5,
        rng_seed: int = None,
    ):
        assert patch_size % 8 == 0, \
            f"patch_size must be divisible by 8, got {patch_size}"

        # Filter images that are too small in PD domain
        self.pd_images = []
        for i, pd in enumerate(pd_images):
            _, Hd, Wd = pd.shape
            if Hd >= patch_size and Wd >= patch_size:
                self.pd_images.append(pd)
            else:
                print(f"  [WARNING] PD image #{i} ({Hd}×{Wd}) is smaller than "
                      f"patch_size={patch_size} — skipped for training.")

        if not self.pd_images:
            raise ValueError(
                f"All PD images are smaller than patch_size={patch_size}. "
                "Reduce patch_size or use larger input images."
            )

        self.patch_size      = patch_size
        self.num_patches     = num_patches
        self.neighbor_radius = neighbor_radius
        self.n_masked        = max(1, int(patch_size * patch_size * mask_ratio))
        self.rng             = np.random.default_rng(rng_seed)
        self.shapes          = [(pd.shape[1], pd.shape[2]) for pd in self.pd_images]
        self.n_images        = len(self.pd_images)

    def __len__(self) -> int:
        return self.num_patches

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        P = self.patch_size

        # Pick a random PD image, then a random spatial patch
        img_idx = int(self.rng.integers(0, self.n_images))
        Hd, Wd  = self.shapes[img_idx]
        r0      = int(self.rng.integers(0, Hd - P))
        c0      = int(self.rng.integers(0, Wd - P))

        target    = self.pd_images[img_idx][:, r0:r0 + P, c0:c0 + P].copy()  # (r², P, P)
        corrupted = target.copy()
        mask      = np.zeros((P, P), dtype=np.float32)

        corrupted, mask = self._apply_masking(target, corrupted, mask)

        return (
            torch.from_numpy(corrupted),            # (r², P, P)
            torch.from_numpy(target),               # (r², P, P)
            torch.from_numpy(mask).unsqueeze(0),    # (1,  P, P) — broadcasts over r²
        )

    def _apply_masking(
        self,
        target:    np.ndarray,
        corrupted: np.ndarray,
        mask:      np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized blind-spot masking; all r² channels masked simultaneously."""
        P   = self.patch_size
        rad = self.neighbor_radius

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

        corrupted[:, rows, cols] = target[:, nr, nc]   # all r² channels
        mask[rows, cols]         = 1.0

        return corrupted, mask


# ============================================================
# 5. Training
# ============================================================

def train_apbsn_multi(
    model: nn.Module,
    images: List[np.ndarray],           # normalized [0,1] float32 arrays
    pd_stride: int = 2,
    patch_size: int = 64,
    batch_size: int = 64,
    num_epochs: int = 100,
    learning_rate: float = 4e-4,
    val_percentage: float = 0.1,
    device: torch.device = None,
) -> nn.Module:
    """
    Train AP-BSN jointly on multiple images (one training run for all).

    All images are PD-transformed on CPU before dataset creation.
    Patches are sampled uniformly across the entire PD-image pool.
    Loss is MSE at masked spatial positions only.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    r  = pd_stride
    nc = r * r

    # Pre-compute PD transform for every training image (CPU, once)
    print(f"\nPre-computing PD transforms (stride r={r}, {nc} channels)...")
    pd_images = []
    for i, img in enumerate(images):
        pd = _numpy_pd(img, r)
        _, Hd, Wd = pd.shape
        print(f"  image #{i}: {img.shape} → PD {Hd}×{Wd}×{nc}")
        pd_images.append(pd)

    # Auto-correct patch_size based on smallest PD image
    min_pd_dim = min(min(pd.shape[1], pd.shape[2]) for pd in pd_images)
    if patch_size > min_pd_dim:
        adjusted = (min_pd_dim // 8) * 8
        print(f"  WARNING: patch_size={patch_size} > smallest PD dim {min_pd_dim}. "
              f"Auto-adjusting to {adjusted}.")
        patch_size = adjusted

    # Scale patches/epoch with image count (plateau for small sets)
    patches_per_epoch = max(2000, 500 * len(images))
    n_val   = max(1, int(patches_per_epoch * val_percentage))
    n_train = patches_per_epoch - n_val

    train_ds = MultiImageAPBSNDataset(pd_images, patch_size, n_train, rng_seed=42)
    val_ds   = MultiImageAPBSNDataset(pd_images, patch_size, n_val,   rng_seed=99)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )
    loss_fn = nn.MSELoss(reduction='sum')

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nDevice: {device}  |  Parameters: {n_params:,}")
    print(f"Training on {len(train_ds.pd_images)} image(s)")
    print(f"pd_stride={r}  patch_size={patch_size}  batch_size={batch_size}  epochs={num_epochs}")
    print(f"Patches/epoch: train={n_train}  val={n_val}\n")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        model.train()
        tr_loss, tr_count = 0.0, 0
        for noisy_in, clean_tgt, mask in train_loader:
            noisy_in  = noisy_in.to(device)
            clean_tgt = clean_tgt.to(device)
            mask      = mask.to(device)          # (B, 1, P, P) — broadcasts over r²

            optimizer.zero_grad()
            pred = model(noisy_in)
            loss = loss_fn(pred * mask, clean_tgt * mask)
            loss.backward()
            optimizer.step()

            tr_loss  += loss.item()
            tr_count += mask.sum().item() * nc

        model.eval()
        vl_loss, vl_count = 0.0, 0
        with torch.no_grad():
            for noisy_in, clean_tgt, mask in val_loader:
                noisy_in  = noisy_in.to(device)
                clean_tgt = clean_tgt.to(device)
                mask      = mask.to(device)
                pred      = model(noisy_in)
                vl_loss  += loss_fn(pred * mask, clean_tgt * mask).item()
                vl_count += mask.sum().item() * nc

        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch:3d}/{num_epochs}]  "
                  f"train={tr_loss / max(tr_count, 1):.6f}  "
                  f"val={vl_loss / max(vl_count, 1):.6f}  "
                  f"time={time.time() - t0:.1f}s")

    print("\nTraining complete.")
    return model


# ============================================================
# 6. Inference
# ============================================================

def _run_bsn_on_pd(
    model:  nn.Module,
    pd:     torch.Tensor,   # (1, r², Hd, Wd) on device
    device: torch.device,
) -> torch.Tensor:
    """Forward pass through BSN with U-Net alignment padding (multiple of 8)."""
    _, _, Hd, Wd = pd.shape
    unet_ph = (8 - Hd % 8) % 8
    unet_pw = (8 - Wd % 8) % 8
    if unet_ph > 0 or unet_pw > 0:
        pd = F.pad(pd, (0, unet_pw, 0, unet_ph), mode='reflect')
    with torch.no_grad():
        out = model(pd)
    if unet_ph > 0 or unet_pw > 0:
        out = out[:, :, :Hd, :Wd]
    return out


def predict_apbsn(
    model:      nn.Module,
    image:      np.ndarray,    # (H, W) float32 [0, 1]
    pd_stride:  int,
    device:     torch.device,
    avg_shifts: bool = True,
) -> np.ndarray:
    """
    Run AP-BSN inference on a single image.

    avg_shifts=False  Single PD pass — fast, may show faint PD-grid texture.
    avg_shifts=True   Average over all r² shift alignments — removes grid
                      artifacts (r=2: 4 passes, r=5: 25 passes).
    """
    model.eval()
    r = pd_stride
    H, W = image.shape

    if not avg_shifts:
        img_t      = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)
        pd, pad_hw = pd_downsample(img_t, r)
        pd_out     = _run_bsn_on_pd(model, pd, device)
        denoised   = pd_upsample(pd_out.cpu(), r, pad_hw, (H, W))
        return np.clip(denoised.squeeze().numpy(), 0.0, 1.0).astype(np.float32)

    # AP quality mode: average over all r² shift alignments
    output_sum = np.zeros((H, W), dtype=np.float64)
    count_sum  = np.zeros((H, W), dtype=np.float64)
    total      = r * r
    pass_idx   = 0

    for p in range(r):
        for q in range(r):
            pass_idx += 1
            crop_np = image[p:, q:]
            ch, cw  = crop_np.shape

            crop_t     = torch.from_numpy(crop_np.copy()).unsqueeze(0).unsqueeze(0).to(device)
            pd, pad_hw = pd_downsample(crop_t, r)
            pd_out     = _run_bsn_on_pd(model, pd, device)
            result     = pd_upsample(pd_out.cpu(), r, pad_hw, (ch, cw))
            result_np  = result.squeeze().numpy()

            ph = result_np.shape[0]
            pw = result_np.shape[1]
            output_sum[p:p + ph, q:q + pw] += result_np.astype(np.float64)
            count_sum [p:p + ph, q:q + pw] += 1.0
            print(f"    AP pass: {pass_idx}/{total}  (shift p={p}, q={q})")

    return np.clip(
        (output_sum / np.maximum(count_sum, 1.0)).astype(np.float32),
        0.0, 1.0,
    )


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
    denoised_orig = (denoised * (img_max - img_min) + img_min).astype(np.float32)
    tifffile.imwrite(tif_path, denoised_orig)
    print(f"  Saved TIF: {tif_path}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image,    cmap='gray'); axes[0].set_title('Original');       axes[0].axis('off')
    axes[1].imshow(denoised, cmap='gray'); axes[1].set_title('AP-BSN Denoised'); axes[1].axis('off')
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
        description="AP-BSN multi-image SEM denoiser: train once, denoise all."
    )
    parser.add_argument('--input_dir',    type=str, default='.',
                        help='Directory with images to denoise (also used for training '
                             'unless --train_dir is specified)')
    parser.add_argument('--train_dir',    type=str, default='',
                        help='Optional: directory of images used ONLY for training. '
                             'All images in --input_dir will still be denoised.')
    parser.add_argument('--output_dir',   type=str, default='denoised',
                        help='Directory for denoised outputs')
    parser.add_argument('--pd_stride',    type=int, default=2,
                        help='PD stride r: 2=SEM pixel-indep. noise, 5=camera sRGB')
    parser.add_argument('--patch_size',   type=int, default=64,
                        help='Patch size in PD domain (must be divisible by 8)')
    parser.add_argument('--batch_size',   type=int, default=64)
    parser.add_argument('--epochs',       type=int, default=100)
    parser.add_argument('--avg_shifts',   action='store_true', default=True,
                        help='AP quality mode: average over all r² shift alignments (default: on)')
    parser.add_argument('--fast',         action='store_true', default=False,
                        help='Single-pass inference — faster but may show faint PD-grid texture')
    parser.add_argument('--save_model',   type=str, default='',
                        help='Path to save trained model weights (.pt)')
    parser.add_argument('--load_model',   type=str, default='',
                        help='Path to load pre-trained weights — skips training entirely')
    args = parser.parse_args()

    avg_shifts = not args.fast
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── 1. Discover images ────────────────────────────────────────────────────
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

    # ── 2. Load training images ───────────────────────────────────────────────
    print("\nLoading training images...")
    train_images = []
    for p in train_paths:
        img, img_min, img_max = load_sem_image(str(p))
        train_images.append(img)
        print(f"  {p.name}: shape={img.shape}  range=[{img_min:.1f}, {img_max:.1f}]")

    # ── 3. Build model ────────────────────────────────────────────────────────
    model = BSNUNet(pd_stride=args.pd_stride, base_features=32)

    if args.load_model and os.path.isfile(args.load_model):
        model.load_state_dict(torch.load(args.load_model, map_location=device))
        model = model.to(device)
        print(f"\nLoaded pre-trained weights: {args.load_model}  (skipping training)")
    else:
        # ── 4. Train ONCE on all training images ──────────────────────────────
        model = train_apbsn_multi(
            model, train_images,
            pd_stride=args.pd_stride,
            patch_size=args.patch_size,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            device=device,
        )
        if args.save_model:
            torch.save(model.state_dict(), args.save_model)
            print(f"Model weights saved: {args.save_model}")

    # ── 5. Load inference images ──────────────────────────────────────────────
    print("\nLoading inference images...")
    infer_images = []
    infer_meta   = []
    for p in infer_paths:
        img, img_min, img_max = load_sem_image(str(p))
        infer_images.append(img)
        infer_meta.append((img_min, img_max))
        print(f"  {p.name}: shape={img.shape}")

    # ── 6. Inference on every image ───────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mode_str = (f"AP quality ({args.pd_stride}²={args.pd_stride**2} passes)"
                if avg_shifts else "single-pass (fast)")
    print(f"\nRunning AP-BSN inference [{mode_str}] on {len(infer_paths)} image(s)...")

    for i, (p, img, (img_min, img_max)) in enumerate(
            zip(infer_paths, infer_images, infer_meta)):
        print(f"\n[{i+1}/{len(infer_paths)}] {p.name}")
        denoised = predict_apbsn(
            model, img,
            pd_stride=args.pd_stride,
            device=device,
            avg_shifts=avg_shifts,
        )
        tif_path = str(out_dir / f"{p.stem}_apbsn_denoised.tif")
        png_path = str(out_dir / f"{p.stem}_apbsn_comparison.png")
        save_outputs(img, denoised, img_min, img_max, tif_path, png_path)

    print(f"\nDone. All results saved to '{out_dir}/'")


if __name__ == '__main__':
    main()

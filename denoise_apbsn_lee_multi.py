# ============================================================
# SEM Image Denoising — AP-BSN (Lee et al., CVPR 2022)
# Official-style PyTorch port — Multi-Image Version
# ============================================================
# Paper:  "AP-BSN: Self-Supervised Denoising for Real-World Images
#          via Asymmetric PD and Blind-Spot Network"
#          Lee et al., CVPR 2022
# Source: https://github.com/wooseoklee4/AP-BSN
#
# Train ONE shared APBSN on a pool of images, then denoise
# every image with the same model.
#
# Differences from denoise_apbsn_lee.py (single-image)
# -----------------------------------------------------
#   MultiImagePatchDataset:
#     pools raw crops from ALL training images;
#     PD is still applied inside APBSN.forward() — no pre-computation
#   patches_per_epoch scales with image count (500×N, capped 2000–6000)
#   --input_dir / --output_dir / --train_dir  (folder-level CLI)
#   Per-image output: {stem}_denoised.tif + {stem}_comparison.png
#
# Identical to denoise_apbsn_lee.py:
#   = APBSN wrapper (PD inside forward, APBSN.denoise for R3)
#   = pixel_shuffle_down/up_sampling (official util.py)
#   = DBSNl architecture with corrected DC_branchl layer order
#     CMConv → ReLU → [1×1+ReLU]×2 → DCl×N → 1×1+ReLU
#   = L1 loss on all pixels (self_L1)
#   = Adam (lr=1e-4, betas=(0.9,0.999)) + StepLR
#   = R3 via APBSN.denoise() (torch, no PD in refinement passes)
#
# Usage
# -----
#   python denoise_apbsn_lee_multi.py \
#       --input_dir ./sem_images --output_dir ./denoised
#
#   # Camera sRGB noise (asymmetric PD):
#   python denoise_apbsn_lee_multi.py \
#       --input_dir ./sem_images --output_dir ./denoised \
#       --pd_a 5 --pd_b 2 --pd_pad 2 --base_ch 128
#
#   # Train on representative subset, denoise all:
#   python denoise_apbsn_lee_multi.py \
#       --train_dir ./train_imgs --input_dir ./all_imgs \
#       --output_dir ./denoised --save_model apbsn_lee.pth
#
#   # Load pre-trained, skip training:
#   python denoise_apbsn_lee_multi.py \
#       --input_dir ./all_imgs --output_dir ./denoised \
#       --load_model apbsn_lee.pth
# ============================================================
# Requirements: torch>=2.0  tifffile  matplotlib  numpy
# ============================================================

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
# 1. Image Loading & Discovery
# ============================================================

def load_sem_image(path: str) -> Tuple[np.ndarray, float, float]:
    """Load SEM image, normalize to float32 [0, 1] grayscale."""
    img = tifffile.imread(path).astype(np.float32)
    if img.ndim == 3 and img.shape[-1] in (3, 4):
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
# 2. Pixel-Shuffle PD Utilities  (src/util/util.py in official repo)
# ============================================================

def pixel_shuffle_down_sampling(
    x:         torch.Tensor,
    f:         int,
    pad:       int   = 0,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """
    Pixel-Shuffle Downsampling (official util.py).

    Rearranges pixels so that neighbors within f×f blocks become distant,
    decorrelating spatially-correlated noise.  With pad=0 the output has
    the same spatial size as the input.

    (B, C, H, W) → (B, C, H + 2·f·pad, W + 2·f·pad)
    Requires H, W divisible by f.
    """
    b, c, H, W = x.shape
    unshuffled = F.pixel_unshuffle(x, f)   # (B, C·f², H//f, W//f)
    if pad != 0:
        unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
    Hp = H // f + 2 * pad
    Wp = W // f + 2 * pad
    return (unshuffled
            .view(b, c, f, f, Hp, Wp)
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(b, c, f * Hp, f * Wp))


def pixel_shuffle_up_sampling(
    x:   torch.Tensor,
    f:   int,
    pad: int = 0,
) -> torch.Tensor:
    """
    Inverse of pixel_shuffle_down_sampling (official util.py).

    (B, C, H + 2·f·pad, W + 2·f·pad) → (B, C, H, W)
    """
    b, c, H, W = x.shape
    before_shuffle = (x
                      .view(b, c, f, H // f, f, W // f)
                      .permute(0, 1, 2, 4, 3, 5)
                      .reshape(b, c * f * f, H // f, W // f))
    if pad != 0:
        before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
    return F.pixel_shuffle(before_shuffle, f)


# ============================================================
# 3. DBSNl Architecture  (src/model/DBSNl.py in official repo)
# ============================================================

class CentralMaskedConv2d(nn.Conv2d):
    """
    Conv2d with center kernel weight permanently zeroed (architectural blind-spot).
    Output at (i, j) cannot depend on input at (i, j), at both train and eval time.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        kH, kW = self.kernel_size
        self.register_buffer('_cmask', torch.ones(1, 1, kH, kW))
        self._cmask[:, :, kH // 2, kW // 2] = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x,
            self.weight * self._cmask,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class DCl(nn.Module):
    """Dilated residual block: dilated 3×3 → ReLU → 1×1, + skip."""
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3,
                      padding=dilation, dilation=dilation, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class DC_branchl(nn.Module):
    """
    One dilated branch of DBSNl (stride ∈ {2, 3}).

    Official layer order (DBSNl.py):
      CentralMaskedConv → ReLU → [1×1 Conv + ReLU] × 2 → DCl × N → 1×1 Conv + ReLU
    """
    def __init__(self, stride: int, base_ch: int, num_module: int):
        super().__init__()
        kernel  = 2 * stride - 1
        padding = kernel // 2
        layers  = [
            CentralMaskedConv2d(base_ch, base_ch,
                                kernel_size=kernel, padding=padding, bias=True),
            nn.ReLU(inplace=True),          # bare ReLU immediately after CMConv (official)
            nn.Conv2d(base_ch, base_ch, 1, bias=True), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, 1, bias=True), nn.ReLU(inplace=True),
        ]
        for _ in range(num_module):
            layers.append(DCl(base_ch, dilation=stride))
        layers += [nn.Conv2d(base_ch, base_ch, 1, bias=True), nn.ReLU(inplace=True)]
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class DBSNl(nn.Module):
    """
    Dilated Blind-Spot Network (DBSNl) — core BSN used in AP-BSN.

    head  : 1×1 Conv + ReLU
    branch1: DC_branchl(stride=2)
    branch2: DC_branchl(stride=3)
    tail  : concat → four 1×1 projections → output
    """
    def __init__(self, in_ch: int = 1, base_ch: int = 64, num_module: int = 9):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.branch1 = DC_branchl(stride=2, base_ch=base_ch, num_module=num_module)
        self.branch2 = DC_branchl(stride=3, base_ch=base_ch, num_module=num_module)
        self.tail = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch,       1, bias=True), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch,     base_ch // 2,  1, bias=True), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch // 2, base_ch // 2, 1, bias=True), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch // 2, in_ch,         1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h  = self.head(x)
        b1 = self.branch1(h)
        b2 = self.branch2(h)
        return self.tail(torch.cat([b1, b2], dim=1))


# ============================================================
# 4. APBSN Wrapper  (src/model/APBSN.py in official repo)
# ============================================================

class APBSN(nn.Module):
    """
    AP-BSN wrapper: DBSNl + asymmetric PD + R3 refinement.

    forward(img, pd=pd_a)  — training path
        pixel_shuffle_down(pd) → BSN → pixel_shuffle_up(pd)
        crop must have dims divisible by pd

    denoise(x)  — inference path (official logic)
        1. pad x to be divisible by pd_b
        2. first pass: img_pd_bsn = forward(x, pd=pd_b)
        3. R3 passes: bsn(mix) where mix = mostly-denoised + R3_p noisy pixels
        4. average all R3_T outputs
    """
    def __init__(
        self,
        pd_a:       int   = 2,
        pd_b:       int   = 2,
        pd_pad:     int   = 0,
        R3:         bool  = True,
        R3_T:       int   = 8,
        R3_p:       float = 0.16,
        in_ch:      int   = 1,
        base_ch:    int   = 64,
        num_module: int   = 9,
    ):
        super().__init__()
        self.pd_a   = pd_a
        self.pd_b   = pd_b
        self.pd_pad = pd_pad
        self.R3     = R3
        self.R3_T   = R3_T
        self.R3_p   = R3_p
        self.bsn    = DBSNl(in_ch=in_ch, base_ch=base_ch, num_module=num_module)

    def forward(self, img: torch.Tensor, pd: Optional[int] = None) -> torch.Tensor:
        """PD-down → BSN → PD-up.  Uses pd_a by default (training path)."""
        if pd is None:
            pd = self.pd_a
        if pd > 1:
            pd_img = pixel_shuffle_down_sampling(img, f=pd, pad=self.pd_pad)
        else:
            p = self.pd_pad
            pd_img = F.pad(img, (p, p, p, p))

        pd_denoised = self.bsn(pd_img)

        if pd > 1:
            out = pixel_shuffle_up_sampling(pd_denoised, f=pd, pad=self.pd_pad)
        else:
            p = self.pd_pad
            out = pd_denoised[:, :, p:-p, p:-p] if p > 0 else pd_denoised
        return out

    @torch.no_grad()
    def denoise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full inference with optional R3 refinement (official logic).
        Handles padding so spatial dims are divisible by pd_b.
        """
        b, c, h, w = x.shape
        pad_h = (self.pd_b - h % self.pd_b) % self.pd_b
        pad_w = (self.pd_b - w % self.pd_b) % self.pd_b
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        img_pd_bsn = self.forward(x, pd=self.pd_b)

        if not self.R3:
            return img_pd_bsn[:, :, :h, :w]

        denoised = torch.empty(*x.shape, self.R3_T, device=x.device)
        for t in range(self.R3_T):
            mask = torch.rand_like(x) < self.R3_p
            tmp  = img_pd_bsn.clone()
            tmp[mask] = x[mask]

            p = self.pd_pad
            if p > 0:
                tmp = F.pad(tmp, (p, p, p, p), mode='reflect')
                denoised[..., t] = self.bsn(tmp)[:, :, p:-p, p:-p]
            else:
                denoised[..., t] = self.bsn(tmp)

        return torch.mean(denoised, dim=-1)[:, :, :h, :w]


# ============================================================
# 5. Multi-Image Training Dataset
# ============================================================

class MultiImagePatchDataset(Dataset):
    """
    Multi-image patch dataset for APBSN training.

    Stores all training images in memory.  Each sample is a raw crop
    from a randomly chosen image — no PD pre-computation.
    PD is applied inside APBSN.forward() during training.

    crop_size must be divisible by pd_a.
    Images smaller than crop_size in either dimension are skipped.

    Parameters
    ----------
    images      : list of (H_i, W_i) float32 [0, 1] noisy images
    crop_size   : patch height and width (must be divisible by pd_a)
    num_patches : virtual epoch length
    rng_seed    : numpy seed for reproducibility
    """
    def __init__(
        self,
        images:      List[np.ndarray],
        crop_size:   int   = 64,
        num_patches: int   = 2000,
        rng_seed:    Optional[int] = None,
    ):
        self.P           = crop_size
        self.num_patches = num_patches
        self.rng         = np.random.default_rng(rng_seed)

        # Filter out images too small to crop from
        self.images: List[np.ndarray] = []
        skipped = 0
        for i, img in enumerate(images):
            H, W = img.shape
            if H < crop_size or W < crop_size:
                print(f"  [WARNING] Image #{i} ({H}×{W}) < crop_size={crop_size} "
                      f"— skipped for training.")
                skipped += 1
                continue
            self.images.append(img)

        if not self.images:
            raise ValueError(
                f"All {len(images)} images are smaller than crop_size={crop_size}. "
                f"Reduce --crop_size."
            )
        if skipped:
            print(f"  {len(self.images)}/{len(images)} images used for training "
                  f"({skipped} skipped — too small for crop_size={crop_size}).")

    def __len__(self) -> int:
        return self.num_patches

    def __getitem__(self, _: int) -> torch.Tensor:
        P   = self.P
        img = self.images[self.rng.integers(0, len(self.images))]
        H, W = img.shape
        r0  = self.rng.integers(0, H - P + 1)
        c0  = self.rng.integers(0, W - P + 1)
        patch = img[r0:r0 + P, c0:c0 + P][np.newaxis].copy()   # (1, P, P)
        return torch.from_numpy(patch)


# ============================================================
# 6. Training
# ============================================================

def train_apbsn(
    model:         APBSN,
    images:        List[np.ndarray],
    crop_size:     int   = 64,
    batch_size:    int   = 32,
    num_epochs:    int   = 100,
    learning_rate: float = 1e-4,
    device:        Optional[torch.device] = None,
) -> APBSN:
    """
    Train AP-BSN on a pool of noisy images.

    Loss (official self_L1):
        L = ‖APBSN.forward(crop, pd=pd_a) − crop‖₁

    patches_per_epoch scales with image count:
        clamp(500 × N, 2000, 6000)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    pd_a = model.pd_a
    if crop_size % pd_a != 0:
        crop_size = (crop_size // pd_a) * pd_a
        print(f"INFO: crop_size rounded to {crop_size} (must be divisible by pd_a={pd_a}).")

    n_total  = int(min(6000, max(2000, 500 * len(images))))
    n_val    = max(1, n_total // 10)
    n_train  = n_total - n_val

    train_ds = MultiImagePatchDataset(images, crop_size, n_train, rng_seed=42)
    val_ds   = MultiImagePatchDataset(images, crop_size, n_val,   rng_seed=99)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0)

    optimizer   = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    step_epochs = max(1, num_epochs // 3)
    scheduler   = optim.lr_scheduler.StepLR(optimizer, step_size=step_epochs, gamma=0.1)
    loss_fn     = nn.L1Loss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}  |  Parameters: {n_params:,}")
    print(f"Training on {len(train_ds.images)} image(s) — "
          f"pd_a={pd_a}  pd_b={model.pd_b}  pd_pad={model.pd_pad}")
    print(f"crop_size={crop_size}  batch_size={batch_size}  epochs={num_epochs}")
    print(f"Patches/epoch: train={n_train}  val={n_val}")
    print(f"R3 at inference: {'enabled' if model.R3 else 'disabled'} "
          f"(T={model.R3_T}, p={model.R3_p})")

    pix_train = n_train * crop_size * crop_size
    pix_val   = n_val   * crop_size * crop_size

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        model.train()
        tr_loss = 0.0
        for batch in train_loader:
            batch  = batch.to(device)
            output = model(batch)              # APBSN.forward() with pd=pd_a
            loss   = loss_fn(output, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * batch.numel()

        model.eval()
        vl_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch   = batch.to(device)
                vl_loss += loss_fn(model(batch), batch).item() * batch.numel()

        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch:3d}/{num_epochs}]  "
                  f"train={tr_loss / pix_train:.6f}  "
                  f"val={vl_loss / pix_val:.6f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                  f"{time.time() - t0:.1f}s")

    print("Training complete.")
    return model


# ============================================================
# 7. Per-Image Inference
# ============================================================

def predict(
    model:  APBSN,
    image:  np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """
    Run full AP-BSN inference for one image via APBSN.denoise().

    Converts (H, W) → (1, 1, H, W) tensor, calls model.denoise()
    which handles: pd_b padding, first-pass PD→BSN→PD⁻¹, and R3.
    """
    x   = torch.from_numpy(image[np.newaxis, np.newaxis]).to(device)  # (1,1,H,W)
    out = model.denoise(x)   # @torch.no_grad() is on denoise()
    return np.clip(out.cpu().squeeze().numpy(), 0.0, 1.0).astype(np.float32)


# ============================================================
# 8. Per-Image Save
# ============================================================

def save_outputs(
    image:    np.ndarray,
    denoised: np.ndarray,
    img_min:  float,
    img_max:  float,
    tif_path: str,
    png_path: str,
) -> None:
    """Save denoised TIF (original value range) and 3-panel comparison PNG."""
    denoised_orig = (denoised * (img_max - img_min) + img_min).astype(np.float32)
    tifffile.imwrite(tif_path, denoised_orig)
    print(f"  Saved TIF: {tif_path}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image,    cmap='gray'); axes[0].set_title('Original SEM');   axes[0].axis('off')
    axes[1].imshow(denoised, cmap='gray'); axes[1].set_title('AP-BSN (Lee)');   axes[1].axis('off')
    diff = np.abs(image - denoised) * 3
    axes[2].imshow(diff,     cmap='hot');  axes[2].set_title('Difference (×3)');axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  Saved PNG: {png_path}")


# ============================================================
# 9. Main Pipeline
# ============================================================

def main() -> None:
    """
    Full multi-image AP-BSN (Lee et al. style) pipeline:
      load → build APBSN → train(pd_a, L1) → per-image denoise(pd_b, R3) → save

    Parameter quick-reference
    -------------------------
    Scenario               pd_a  pd_b  pd_pad  base_ch  epochs
    SEM pixel-independent  2     2     0       64       100     (default)
    Camera sRGB noise      5     2     2       128      100
    Fast / preview         2     2     0       32       50
    Low GPU RAM            2     2     0       32       100
    """
    parser = argparse.ArgumentParser(
        description="AP-BSN (Lee et al., CVPR 2022) official-style — multi-image"
    )
    parser.add_argument('--input_dir',   type=str,   default='.',
                        help='Directory with input images (train + inference unless '
                             '--train_dir is set)')
    parser.add_argument('--train_dir',   type=str,   default='',
                        help='Optional: separate directory used ONLY for training. '
                             'All images in --input_dir will still be denoised.')
    parser.add_argument('--output_dir',  type=str,   default='denoised',
                        help='Directory to write denoised results')
    # PD parameters
    parser.add_argument('--pd_a',        type=int,   default=2,
                        help='PD stride for TRAINING  (paper: 5 for camera; 2 for SEM)')
    parser.add_argument('--pd_b',        type=int,   default=2,
                        help='PD stride for INFERENCE (paper: 2)')
    parser.add_argument('--pd_pad',      type=int,   default=0,
                        help='PD padding (paper: 2 for camera; 0 for SEM)')
    # DBSNl parameters
    parser.add_argument('--base_ch',     type=int,   default=64,
                        help='DBSNl base channels (paper: 128; SEM default: 64)')
    parser.add_argument('--num_module',  type=int,   default=9,
                        help='DCl residual blocks per branch (paper: 9)')
    # Training parameters
    parser.add_argument('--crop_size',   type=int,   default=64,
                        help='Training crop size (must be divisible by pd_a)')
    parser.add_argument('--batch_size',  type=int,   default=32)
    parser.add_argument('--epochs',      type=int,   default=100)
    parser.add_argument('--lr',          type=float, default=1e-4,
                        help='Initial learning rate (paper: 1e-4)')
    # R3 parameters
    parser.add_argument('--R3_T',        type=int,   default=8,
                        help='R3 refinement passes (paper: 8)')
    parser.add_argument('--R3_p',        type=float, default=0.16,
                        help='R3 random replacement probability (paper: 0.16)')
    parser.add_argument('--no_r3',       action='store_true',
                        help='Disable R3 (faster single-pass inference)')
    # Model persistence
    parser.add_argument('--save_model',  type=str,   default='',
                        help='Save trained model weights to this path (.pth)')
    parser.add_argument('--load_model',  type=str,   default='',
                        help='Load weights from this path — skips training')
    # Hardware
    parser.add_argument('--device',      type=str,   default=None,
                        help='Device override: cuda, cpu, cuda:1 … (default: auto)')
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device
        else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    os.makedirs(args.output_dir, exist_ok=True)

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

    # ── 2. Build APBSN ────────────────────────────────────────────────────────
    model = APBSN(
        pd_a       = args.pd_a,
        pd_b       = args.pd_b,
        pd_pad     = args.pd_pad,
        R3         = not args.no_r3,
        R3_T       = args.R3_T,
        R3_p       = args.R3_p,
        in_ch      = 1,
        base_ch    = args.base_ch,
        num_module = args.num_module,
    )

    if args.pd_a == args.pd_b:
        print(f"\nSymmetric PD (pd_a = pd_b = {args.pd_a}): "
              f"suitable for SEM pixel-independent noise.")
    else:
        print(f"\nAsymmetric PD (pd_a={args.pd_a} train, pd_b={args.pd_b} infer): "
              f"suited for spatially-correlated camera noise.")

    # ── 3. Train or load ──────────────────────────────────────────────────────
    if args.load_model and os.path.isfile(args.load_model):
        ckpt = torch.load(args.load_model, map_location=device)
        model.load_state_dict(ckpt)
        model = model.to(device)
        print(f"Loaded model from '{args.load_model}' — skipping training.")
    else:
        print(f"\nLoading training images...")
        train_images: List[np.ndarray] = []
        for p in train_paths:
            img, img_min, img_max = load_sem_image(str(p))
            train_images.append(img)
            print(f"  {p.name}: {img.shape}  range=[{img_min:.1f}, {img_max:.1f}]")

        model = train_apbsn(
            model, train_images,
            crop_size     = args.crop_size,
            batch_size    = args.batch_size,
            num_epochs    = args.epochs,
            learning_rate = args.lr,
            device        = device,
        )

        if args.save_model:
            os.makedirs(os.path.dirname(args.save_model) or ".", exist_ok=True)
            torch.save(model.state_dict(), args.save_model)
            print(f"Model saved: {args.save_model}")

    # ── 4. Per-image inference + save ─────────────────────────────────────────
    mode_str = f"pd_b={args.pd_b}"
    if not args.no_r3:
        mode_str += f" + R3(T={args.R3_T}, p={args.R3_p})"
    print(f"\nDenoising {len(infer_paths)} image(s) [{mode_str}]...")

    model.eval()
    for idx, p in enumerate(infer_paths, 1):
        print(f"\n[{idx}/{len(infer_paths)}] {p.name}")
        img, img_min, img_max = load_sem_image(str(p))

        denoised = predict(model, img, device)
        print(f"  Denoised range: [{denoised.min():.3f}, {denoised.max():.3f}]")

        stem     = p.stem
        tif_path = os.path.join(args.output_dir, f"{stem}_denoised.tif")
        png_path = os.path.join(args.output_dir, f"{stem}_comparison.png")
        save_outputs(img, denoised, img_min, img_max, tif_path, png_path)

    print(f"\nAll done. Results in: {args.output_dir}/")


if __name__ == '__main__':
    main()

# ============================================================
# SEM Image Denoising — AP-BSN Faithful, Multi-Image
# ============================================================
# Paper: "AP-BSN: Self-Supervised Denoising for Real-World Images
#         via Asymmetric PD and Blind-Spot Network"
#         Lee et al., CVPR 2022
# arXiv:  https://arxiv.org/abs/2203.11799
# GitHub: https://github.com/wooseoklee4/AP-BSN
#
# Train ONCE on a pool of images, then denoise every image with the
# same model. The shared model sees more diverse content → richer
# noise statistics → better generalisation.
#
# Differences from denoise_apbsn_faithful.py (single-image):
#   + MultiImageAPBSNDataset: pools PD sub-images from ALL training
#     images; patches drawn uniformly across the whole pool
#   + patches_per_epoch scales with image count (max 4000, min 2000)
#   + --train_dir: optional separate subset for training
#   + --save_model / --load_model checkpoint support
#   + Per-image output: {stem}_denoised.tif + {stem}_comparison.png
#   + R3 per-image (uses each image's own noisy pixels as replacement pool)
#
# Identical to denoise_apbsn_faithful.py:
#   = DBSNl architecture (CentralMaskedConv2d, DCl, DC_branchl)
#   = _numpy_pd / _numpy_pd_inv (PD operations)
#   = L1 loss on ALL pixels (paper Eq. 2)
#   = Asymmetric PD: pd_a (train) vs pd_b (infer)
#   = R3 random-replacing refinement (paper §3.3)
#   = Adam + CosineAnnealingLR, lr=4e-4 → 1e-6
#
# Usage:
#   python denoise_apbsn_faithful_multi.py \
#       --input_dir ./sem_images --output_dir ./denoised
#
#   # Asymmetric PD for camera sRGB noise:
#   python denoise_apbsn_faithful_multi.py \
#       --input_dir ./sem_images --output_dir ./denoised \
#       --pd_a 5 --pd_b 2 --base_ch 128
#
#   # Train on 2 representative images, denoise all:
#   python denoise_apbsn_faithful_multi.py \
#       --train_dir ./train_imgs --input_dir ./all_imgs \
#       --output_dir ./denoised --save_model apbsn_sem.pt
#   python denoise_apbsn_faithful_multi.py \
#       --input_dir ./all_imgs --output_dir ./denoised \
#       --load_model apbsn_sem.pt
# ============================================================
# Requirements: torch>=2.0  tifffile  matplotlib  numpy
# ============================================================

import argparse
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

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
# 1. Image Loading & Discovery
# ============================================================

def load_sem_image(path: str) -> Tuple[np.ndarray, float, float]:
    """Load SEM image, normalize to float32 [0, 1] grayscale."""
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
# 2. PD Operations (numpy, CPU)
# ============================================================

def _numpy_pd(image: np.ndarray, r: int) -> np.ndarray:
    """
    Pixel-Shuffle Downsampling: (H, W) → (r², Hd, Wd).

    Reflect-pads to the nearest multiple of r, then groups pixels by
    (row % r, col % r) phase offset into r² independent sub-images.

        output[ph*r + pq, row, col] = image_padded[row*r + ph, col*r + pq]
    """
    H, W  = image.shape
    pad_h = (r - H % r) % r
    pad_w = (r - W % r) % r
    if pad_h > 0 or pad_w > 0:
        image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
    Hp, Wp = image.shape
    Hd, Wd = Hp // r, Wp // r
    pd = image.reshape(Hd, r, Wd, r).transpose(1, 3, 0, 2).reshape(r * r, Hd, Wd)
    return pd.astype(np.float32)


def _numpy_pd_inv(pd: np.ndarray, r: int, orig_H: int, orig_W: int) -> np.ndarray:
    """
    Inverse PD: (r², Hd, Wd) → (orig_H, orig_W).
    Reverses _numpy_pd; crops reflect-padding via orig_H / orig_W.
    """
    _, Hd, Wd = pd.shape
    img = np.zeros((Hd * r, Wd * r), dtype=np.float32)
    for ch in range(r * r):
        ph, pq = divmod(ch, r)
        img[ph::r, pq::r] = pd[ch]
    return img[:orig_H, :orig_W]


# ============================================================
# 3. DBSNl Architecture
# ============================================================

class CentralMaskedConv2d(nn.Conv2d):
    """
    Conv2d with the center kernel weight permanently zeroed (paper §3.1).

    Creates an architectural blind-spot: the output at (i, j) does not
    depend on the input at (i, j). This holds at both train and eval time,
    unlike N2V-style training-time masking.

    Kernel size per branch: 2*stride - 1  (stride=2 → 3×3; stride=3 → 5×5).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        kH, kW = self.kernel_size
        self.register_buffer('_cmask', torch.ones(1, 1, kH, kW))
        self._cmask[:, :, kH // 2, kW // 2] = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x, self.weight * self._cmask, self.bias,
            self.stride, self.padding, self.dilation, self.groups,
        )


class DCl(nn.Module):
    """Dilated residual block (DCl): dilated 3×3 conv → ReLU → 1×1 conv → + residual."""
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
    One dilated branch of DBSNl (paper §3.1, official DBSNl.py).

    Sequence (stride ∈ {2, 3}):
      CentralMaskedConv2d(kernel=2*stride-1)  ← blind-spot entry
      → 1×1 Conv + ReLU  ×3
      → DCl(dilation=stride) × num_module
      → 1×1 Conv + ReLU
    """
    def __init__(self, stride: int, base_ch: int, num_module: int):
        super().__init__()
        kernel  = 2 * stride - 1
        padding = kernel // 2
        layers = []
        layers.append(CentralMaskedConv2d(base_ch, base_ch,
                                          kernel_size=kernel, padding=padding, bias=True))
        for _ in range(3):
            layers.append(nn.Conv2d(base_ch, base_ch, 1, bias=True))
            layers.append(nn.ReLU(inplace=True))
        for _ in range(num_module):
            layers.append(DCl(base_ch, dilation=stride))
        layers.append(nn.Conv2d(base_ch, base_ch, 1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class DBSNl(nn.Module):
    """
    Dilated Blind-Spot Network (DBSNl), as used in AP-BSN (CVPR 2022).

    Head    : 1×1 Conv + ReLU
    Branch1 : DC_branchl(stride=2)
    Branch2 : DC_branchl(stride=3)
    Tail    : concat → four 1×1 fusions → output

    Paper defaults: base_ch=128, num_module=9.
    Default here: base_ch=64 (4× faster, slightly lower quality).
    """
    def __init__(self, in_ch: int = 1, base_ch: int = 64, num_module: int = 9):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.branch1 = DC_branchl(stride=2, base_ch=base_ch, num_module=num_module)
        self.branch2 = DC_branchl(stride=3, base_ch=base_ch, num_module=num_module)
        self.tail = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch,       kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch,     base_ch // 2,  kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch // 2, base_ch // 2, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch // 2, in_ch,        kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.head(x)
        return self.tail(torch.cat([self.branch1(h), self.branch2(h)], dim=1))


# ============================================================
# 4. Multi-Image Dataset (pools PD sub-images from all images)
# ============================================================

class MultiImageAPBSNDataset(Dataset):
    """
    Paper-faithful multi-image AP-BSN training dataset.

    Pre-computes PD sub-images (using pd_a) for EVERY training image
    and stores them in a flat pool. Each training sample is a
    single-channel patch drawn uniformly from the pool.

    Loss target = the noisy patch itself (BSN / N2N principle):
        L = ‖DBSNl(patch) − patch‖₁

    No masking needed — CentralMaskedConv2d enforces blind-spot
    architecturally, so the network cannot memorise the center pixel.

    Images whose PD-domain size is smaller than patch_size are skipped
    with a warning.

    Parameters
    ----------
    images      : list of (H_i, W_i) float32 [0, 1] noisy SEM images
    pd_a        : PD stride for training (paper: 5 for camera; 2 for SEM)
    patch_size  : spatial patch size in the PD domain
    num_patches : virtual epoch length
    rng_seed    : numpy seed for reproducibility
    """
    def __init__(
        self,
        images:      List[np.ndarray],
        pd_a:        int = 2,
        patch_size:  int = 64,
        num_patches: int = 2000,
        rng_seed:    int = None,
    ):
        self.P           = patch_size
        self.num_patches = num_patches
        self.rng         = np.random.default_rng(rng_seed)

        # Pre-compute PD sub-images for every training image
        self.pool: List[np.ndarray] = []   # each entry: (r², Hd, Wd)
        skipped = 0
        for i, img in enumerate(images):
            subs = _numpy_pd(img, pd_a)    # (pd_a², Hd, Wd)
            _, Hd, Wd = subs.shape
            if Hd < patch_size or Wd < patch_size:
                print(f"  [WARNING] Image #{i} PD domain {Hd}×{Wd} < "
                      f"patch_size={patch_size} — skipped for training.")
                skipped += 1
                continue
            self.pool.append(subs)

        if not self.pool:
            raise ValueError(
                f"All {len(images)} images produced PD-domain sizes smaller than "
                f"patch_size={patch_size} (pd_a={pd_a}). "
                f"Reduce --patch_size or use smaller --pd_a."
            )
        if skipped:
            print(f"  {len(self.pool)}/{len(images)} images used for training "
                  f"({skipped} skipped).")

        # Build index: (pool_idx, n_subs, Hd, Wd) for fast sampling
        self.meta = [(i, s.shape[0], s.shape[1], s.shape[2])
                     for i, s in enumerate(self.pool)]

    def __len__(self) -> int:
        return self.num_patches

    def __getitem__(self, _idx: int) -> torch.Tensor:
        P  = self.P
        # Uniform random image from pool
        pi = self.rng.integers(0, len(self.pool))
        _, n_s, Hd, Wd = self.meta[pi]
        # Uniform random sub-image and patch location
        si = self.rng.integers(0, n_s)
        r0 = self.rng.integers(0, Hd - P + 1)
        c0 = self.rng.integers(0, Wd - P + 1)
        patch = self.pool[pi][si, r0:r0 + P, c0:c0 + P][np.newaxis].copy()
        return torch.from_numpy(patch)   # (1, P, P)


# ============================================================
# 5. Training (shared model, all images)
# ============================================================

def train_apbsn_multi(
    model:         nn.Module,
    images:        List[np.ndarray],
    pd_a:          int   = 2,
    patch_size:    int   = 64,
    batch_size:    int   = 32,
    num_epochs:    int   = 100,
    learning_rate: float = 4e-4,
    device:        torch.device = None,
) -> nn.Module:
    """
    Train one shared DBSNl on the pooled PD sub-images of all images.

    Patches per epoch scales with image count:
        max(2000, 500 × n_images), capped at 4000.

    Loss (paper Eq. 2): L_BSN = ‖DBSNl(patch) − patch‖₁
    All pixels contribute; no masking needed (architectural blind-spot).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Scale patches with image count
    n_total = min(4000, max(2000, 500 * len(images)))
    n_val   = max(1, n_total // 10)
    n_train = n_total - n_val

    train_ds = MultiImageAPBSNDataset(images, pd_a, patch_size, n_train, rng_seed=42)
    val_ds   = MultiImageAPBSNDataset(images, pd_a, patch_size, n_val,   rng_seed=99)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )
    loss_fn = nn.L1Loss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}  |  Parameters: {n_params:,}")
    print(f"Training on {len(train_ds.pool)} image(s), "
          f"pd_a={pd_a} → {pd_a**2} sub-images each")
    print(f"patch_size={patch_size}  batch_size={batch_size}  epochs={num_epochs}")
    print(f"Patches/epoch: train={n_train}  val={n_val}")

    pix_train = n_train * patch_size * patch_size
    pix_val   = n_val   * patch_size * patch_size

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        model.train()
        tr_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            pred  = model(batch)
            loss  = loss_fn(pred, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * batch.numel()

        model.eval()
        vl_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch    = batch.to(device)
                vl_loss += loss_fn(model(batch), batch).item() * batch.numel()

        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch:3d}/{num_epochs}]  "
                  f"train={tr_loss / pix_train:.6f}  "
                  f"val={vl_loss / pix_val:.6f}  "
                  f"{time.time() - t0:.1f}s")

    print("Training complete.")
    return model


# ============================================================
# 6. Inference (pd_b per image)
# ============================================================

def _infer_single_pass(
    model:  nn.Module,
    image:  np.ndarray,
    pd_b:   int,
    device: torch.device,
) -> np.ndarray:
    """
    One AP-BSN inference pass with stride pd_b (paper §3.2).

    PD(pd_b) → BSN per sub-image (batched) → PD⁻¹ → clip [0, 1].
    Using pd_b ≤ pd_a preserves image structure at inference.
    """
    H, W  = image.shape
    subs  = _numpy_pd(image, pd_b)          # (pd_b², Hd, Wd)
    batch = torch.from_numpy(subs[:, np.newaxis]).to(device)   # (pd_b², 1, Hd, Wd)
    model.eval()
    with torch.no_grad():
        out = model(batch)                   # (pd_b², 1, Hd, Wd)
    out_subs = out.cpu().squeeze(1).numpy()  # (pd_b², Hd, Wd)
    return np.clip(_numpy_pd_inv(out_subs, pd_b, H, W), 0.0, 1.0).astype(np.float32)


def predict_apbsn(
    model:  nn.Module,
    image:  np.ndarray,
    pd_b:   int,
    device: torch.device,
    R3_T:   int   = 8,
    R3_p:   float = 0.16,
    use_r3: bool  = True,
) -> np.ndarray:
    """
    AP-BSN inference with R3 random-replacing refinement (paper §3.3).

    R3 (R3_T=8, R3_p=0.16 from paper / official code):
      D₀ = infer(image)
      for t = 1 … R3_T−1:
          M ~ Bernoulli(R3_p)
          D_t = infer((1−M)·D₀ + M·image)
      output = mean(D₀ … D_{R3_T−1})

    R3 is applied per-image so it always uses the correct noisy source.
    Set use_r3=False for a fast single-pass preview.
    """
    D0 = _infer_single_pass(model, image, pd_b, device)
    if not use_r3:
        return D0

    results = [D0]
    rng = np.random.default_rng(seed=0)
    for t in range(1, R3_T):
        mask  = (rng.random(image.shape) < R3_p).astype(np.float32)
        mixed = (1.0 - mask) * D0 + mask * image
        results.append(_infer_single_pass(model, mixed, pd_b, device))
        print(f"    R3: pass {t}/{R3_T - 1}")

    return np.clip(np.mean(results, axis=0), 0.0, 1.0).astype(np.float32)


# ============================================================
# 7. Per-Image Save
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
    axes[0].imshow(image,    cmap='gray'); axes[0].set_title('Original SEM');        axes[0].axis('off')
    axes[1].imshow(denoised, cmap='gray'); axes[1].set_title('AP-BSN Denoised');     axes[1].axis('off')
    diff = np.abs(image - denoised) * 3
    axes[2].imshow(diff,     cmap='hot');  axes[2].set_title('Difference (×3)');     axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  Saved PNG: {png_path}")


# ============================================================
# 8. Main Pipeline
# ============================================================

def main() -> None:
    """
    Full multi-image AP-BSN pipeline:
      load → train shared DBSNl(pd_a, L1) → per-image infer(pd_b) + R3 → save

    Parameter quick-reference
    -------------------------
    Scenario               pd_a  pd_b  base_ch  epochs  R3
    SEM pixel-independent  2     2     64       100     True  (default)
    Camera sRGB noise      5     2     128      100     True
    Fast / preview         2     2     32       50      False
    Low GPU RAM            2     2     32       100     True
    """
    parser = argparse.ArgumentParser(
        description=(
            "AP-BSN faithful multi-image denoiser: "
            "DBSNl + Asymmetric PD + R3 (CVPR 2022)"
        )
    )
    parser.add_argument('--input_dir',   type=str, default='.',
                        help='Directory with input images (used for both training '
                             'and inference unless --train_dir is set)')
    parser.add_argument('--train_dir',   type=str, default='',
                        help='Optional: separate directory used ONLY for training. '
                             'All images in --input_dir will still be denoised.')
    parser.add_argument('--output_dir',  type=str, default='denoised',
                        help='Directory to write denoised results')
    parser.add_argument('--pd_a',        type=int,   default=2,
                        help='PD stride for TRAINING  (paper: 5 for camera; 2 for SEM)')
    parser.add_argument('--pd_b',        type=int,   default=2,
                        help='PD stride for INFERENCE (paper: 2)')
    parser.add_argument('--base_ch',     type=int,   default=64,
                        help='DBSNl base channels per branch (paper: 128)')
    parser.add_argument('--num_module',  type=int,   default=9,
                        help='DCl residual blocks per branch (paper: 9)')
    parser.add_argument('--patch_size',  type=int,   default=64,
                        help='Training patch size in the PD domain')
    parser.add_argument('--batch_size',  type=int,   default=32)
    parser.add_argument('--epochs',      type=int,   default=100)
    parser.add_argument('--R3_T',        type=int,   default=8,
                        help='R3 refinement passes (paper: 8)')
    parser.add_argument('--R3_p',        type=float, default=0.16,
                        help='R3 random replacement probability (paper: 0.16)')
    parser.add_argument('--no_r3',       action='store_true',
                        help='Disable R3 (faster single-pass inference)')
    parser.add_argument('--save_model',  type=str,   default='',
                        help='Path to save trained model weights (.pt)')
    parser.add_argument('--load_model',  type=str,   default='',
                        help='Path to load pre-trained weights — skips training')
    parser.add_argument('--device',      type=str,   default=None,
                        help='Device override: cuda, cpu, cuda:1 … (default: auto)')
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device
        else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    use_r3 = not args.no_r3
    os.makedirs(args.output_dir, exist_ok=True)

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

    # ── 3. Build model ────────────────────────────────────────────────────────
    model = DBSNl(in_ch=1, base_ch=args.base_ch, num_module=args.num_module)

    # ── 4. Train or load ──────────────────────────────────────────────────────
    if args.load_model and os.path.isfile(args.load_model):
        model.load_state_dict(torch.load(args.load_model, map_location=device))
        model = model.to(device)
        print(f"\nLoaded pre-trained weights: {args.load_model}  (skipping training)")
    else:
        print(f"\nLoading training images...")
        train_images: List[np.ndarray] = []
        for p in train_paths:
            img, img_min, img_max = load_sem_image(str(p))
            train_images.append(img)
            print(f"  {p.name}: shape={img.shape}  "
                  f"range=[{img_min:.1f}, {img_max:.1f}]")

        print(f"\nAsymmetric PD: pd_a={args.pd_a} (train)  "
              f"pd_b={args.pd_b} (infer)")
        if args.pd_a == args.pd_b:
            print("  Symmetric mode — appropriate for SEM pixel-independent noise.")
        else:
            print("  Asymmetric mode — breaks spatial correlation for camera noise.")

        model = train_apbsn_multi(
            model, train_images,
            pd_a        = args.pd_a,
            patch_size  = args.patch_size,
            batch_size  = args.batch_size,
            num_epochs  = args.epochs,
            device      = device,
        )

        if args.save_model:
            torch.save(model.state_dict(), args.save_model)
            print(f"Model saved: {args.save_model}")

    # ── 5. Per-image inference + R3 + save ────────────────────────────────────
    mode_str = f"pd_b={args.pd_b}"
    if use_r3:
        mode_str += f" + R3(T={args.R3_T}, p={args.R3_p})"
    print(f"\nDenoising {len(infer_paths)} image(s) [{mode_str}]...")

    for idx, p in enumerate(infer_paths, 1):
        print(f"\n[{idx}/{len(infer_paths)}] {p.name}")
        img, img_min, img_max = load_sem_image(str(p))

        denoised = predict_apbsn(
            model, img,
            pd_b   = args.pd_b,
            device = device,
            R3_T   = args.R3_T,
            R3_p   = args.R3_p,
            use_r3 = use_r3,
        )
        print(f"  Denoised range: [{denoised.min():.3f}, {denoised.max():.3f}]")

        stem     = p.stem
        tif_path = os.path.join(args.output_dir, f"{stem}_denoised.tif")
        png_path = os.path.join(args.output_dir, f"{stem}_comparison.png")
        save_outputs(img, denoised, img_min, img_max, tif_path, png_path)

    print(f"\nAll done. Results in: {args.output_dir}/")


if __name__ == '__main__':
    main()

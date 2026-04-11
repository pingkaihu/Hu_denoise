# ============================================================
# SEM Image Denoising — AP-BSN (CVPR 2022, pure PyTorch)
# ============================================================
# Paper: "AP-BSN: Self-Supervised Denoising for Real-World Images
#         via Asymmetric PD and Blind-Spot Network"
#         Lee et al., CVPR 2022
# Reference: https://github.com/wooseoklee4/AP-BSN
#
# Core idea — PD (Pixel-Shuffle Downsampling):
#   Given stride r, PD maps (B, 1, H, W) → (B, r², H//r, W//r) by
#   extracting r² spatially-interleaved sub-grids ("phase offsets").
#   Spatially-correlated camera noise becomes approximately pixel-
#   independent in the PD domain — enabling N2V-style blind-spot
#   training at lower resolution without any clean reference images.
#
# Inference modes:
#   avg_shifts=False  Single PD pass → fast, may show faint PD-grid
#   avg_shifts=True   Average over all r² PD shift alignments → removes
#   (default)         grid artifacts; r=2: 4 passes, r=5: 25 passes
#
# Blind-spot: N2V random-neighbour masking (not dilated-conv).
#   SEM noise is pixel-independent, so the approximate blind-spot
#   from masking is sufficient. The PD transform is the key contribution.
#
# pd_stride=2  → SEM (pixel-independent Poisson/Gaussian noise)
# pd_stride=5  → Real camera sRGB (spatially-correlated ISP noise)
#
# Usage:
#   python test_sem.py          # generate test_sem.tif if needed
#   python denoise_apbsn.py     # train + denoise -> denoised_apbsn.tif
# ============================================================
# Requirements: torch>=2.0  tifffile  matplotlib  numpy
# ============================================================

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import time
from typing import Tuple

import numpy as np
import tifffile
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

torch.set_float32_matmul_precision('high')  # use Tensor Core on RTX GPUs


# ============================================================
# 1. Image Loading
# ============================================================

def load_sem_image(path: str) -> Tuple[np.ndarray, float, float]:
    """Load SEM image, normalize to float32 [0, 1] grayscale numpy array.
    Also returns original min/max for restoring pixel values after denoising."""
    img = tifffile.imread(path).astype(np.float32)

    # Convert RGB to grayscale if needed
    if img.ndim == 3 and img.shape[-1] == 3:
        img = img @ np.array([0.2989, 0.5870, 0.1140])

    img_min, img_max = float(img.min()), float(img.max())
    img = (img - img_min) / (img_max - img_min + 1e-8)
    return img, img_min, img_max


# ============================================================
# 2. Pixel-Shuffle Downsampling (PD) Operations
# ============================================================

def pd_downsample(
    x: torch.Tensor,
    stride: int,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Pixel-Shuffle Downsampling: (B, 1, H, W) → (B, r², H//r, W//r).

    Reflects-pads H and W to the nearest multiple of stride r, then
    rearranges pixels so that the r² output channels correspond to the
    r² spatial phase offsets (row % r, col % r) of the input.

    Element mapping:
        pd_out[b, ph*r+pq, row, col] = x[b, 0, row*r+ph, col*r+pq]

    Returns (pd_tensor, (pad_h, pad_w)) — padding sizes needed by pd_upsample.
    """
    B, C, H, W = x.shape
    r = stride
    pad_h = (r - H % r) % r
    pad_w = (r - W % r) % r
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    _, _, Hp, Wp = x.shape
    # (B, C, Hp, Wp) -> (B, C, Hd, r, Wd, r) -> permute -> (B, C*r², Hd, Wd)
    x = x.view(B, C, Hp // r, r, Wp // r, r)
    x = x.permute(0, 1, 3, 5, 2, 4).reshape(B, C * r * r, Hp // r, Wp // r)
    return x, (pad_h, pad_w)


def pd_upsample(
    x: torch.Tensor,
    stride: int,
    pad_hw: Tuple[int, int],   # kept for API clarity; removal is done via orig_hw
    orig_hw: Tuple[int, int],
) -> torch.Tensor:
    """
    Inverse Pixel-Shuffle Upsampling: (B, r², H//r, W//r) → (B, 1, H, W).

    Exactly reverses pd_downsample. orig_hw is the spatial size BEFORE
    any reflect-padding (i.e. the original image H, W), used to crop out
    the padding added by pd_downsample.
    """
    B, _Cr2, Hd, Wd = x.shape
    r = stride
    # (B, r², Hd, Wd) -> (B, 1, r, r, Hd, Wd) -> permute -> (B, 1, H, W)
    x = x.reshape(B, 1, r, r, Hd, Wd)
    x = x.permute(0, 1, 4, 2, 5, 3).reshape(B, 1, Hd * r, Wd * r)
    return x[:, :, :orig_hw[0], :orig_hw[1]]


# ============================================================
# 3. BSN U-Net Architecture
# ============================================================

class DoubleConvBlock(nn.Module):
    """Two sequential Conv2d -> BatchNorm2d -> LeakyReLU(0.1) operations."""

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
    """
    4-level encoder-decoder U-Net for AP-BSN in the PD domain.

    Identical structure to N2VUNet in denoise_torch.py, with the single
    difference that in_channels = out_channels = pd_stride².

    For pd_stride=2: 4 channels.  For pd_stride=5: 25 channels.
    Feature widths (32/64/128/256) are unchanged, so parameter count is
    nearly the same as the single-channel N2V version (~1.8 M params).

    Blind-spot is enforced via N2V-style masking in APBSNDataset —
    NOT by architectural constraints. This is sufficient for SEM images
    because their noise is pixel-independent.

    Input spatial dimensions in the PD domain must be divisible by 8
    (3 MaxPool layers × stride 2). In predict_apbsn() an extra pad/crop
    step handles images where this is not naturally satisfied.
    """

    def __init__(self, pd_stride: int = 2, base_features: int = 32):
        super().__init__()
        nc = pd_stride * pd_stride   # r² PD channels
        f  = base_features           # 32

        # Encoder
        self.enc1 = DoubleConvBlock(nc,    f)        # (B, nc,  H,   W) -> (B, 32,  H,   W)
        self.enc2 = DoubleConvBlock(f,     f * 2)    # -> (B, 64,  H/2, W/2)
        self.enc3 = DoubleConvBlock(f * 2, f * 4)    # -> (B, 128, H/4, W/4)
        self.enc4 = DoubleConvBlock(f * 4, f * 8)    # -> (B, 256, H/8, W/8) bottleneck
        self.pool = nn.MaxPool2d(2)

        # Decoder — upsample + 1×1 conv to halve channels, then skip-cat + DoubleConv
        self.up3  = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(f * 8, f * 4, kernel_size=1),
        )
        self.dec3 = DoubleConvBlock(f * 8, f * 4)   # cat(up3, enc3): 128+128

        self.up2  = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(f * 4, f * 2, kernel_size=1),
        )
        self.dec2 = DoubleConvBlock(f * 4, f * 2)   # cat(up2, enc2): 64+64

        self.up1  = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(f * 2, f, kernel_size=1),
        )
        self.dec1 = DoubleConvBlock(f * 2, f)        # cat(up1, enc1): 32+32

        # Output head — no activation (regression)
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
# 4. AP-BSN Dataset
# ============================================================

class APBSNDataset(Dataset):
    """
    Self-supervised dataset for AP-BSN training in the PD domain.

    The full image is PD-transformed once in __init__ and stored as
    self.pd_image with shape (r², Hd, Wd).  All training patches are
    spatial slices of this pre-computed tensor.

    Masking (N2V-style blind-spot):
      For each selected spatial position (row, col), ALL r² channels are
      simultaneously replaced with the r² values from a randomly chosen
      neighbour position (nr, nc). This prevents the BSN from inferring
      the masked pixel value using the same spatial location in another PD
      phase channel.

      The binary mask has shape (1, P, P) so it broadcasts over (B, r², P, P)
      when computing the MSE loss at masked positions only.

    Parameters
    ----------
    pd_image       : (r², Hd, Wd) float32 — pre-computed PD domain image
    patch_size     : must be divisible by 8 and ≤ min(Hd, Wd)
    num_patches    : virtual epoch length (patches re-sampled each epoch)
    mask_ratio     : fraction of spatial positions masked per patch
    neighbor_radius: max displacement for blind-spot replacement neighbour
    rng_seed       : optional seed for reproducibility
    """

    def __init__(
        self,
        pd_image:        np.ndarray,
        patch_size:      int   = 64,
        num_patches:     int   = 2000,
        mask_ratio:      float = 0.006,
        neighbor_radius: int   = 5,
        rng_seed:        int   = None,
    ):
        assert patch_size % 8 == 0, \
            f"patch_size must be divisible by 8, got {patch_size}"
        _, Hd, Wd = pd_image.shape
        assert Hd >= patch_size and Wd >= patch_size, (
            f"PD-domain image ({Hd}×{Wd}) is too small for patch_size={patch_size}. "
            f"Reduce patch_size (max usable: {(min(Hd, Wd) // 8) * 8}) "
            f"or increase the input image size."
        )

        self.pd_image        = pd_image
        self.patch_size      = patch_size
        self.num_patches     = num_patches
        self.neighbor_radius = neighbor_radius
        self.Hd, self.Wd    = Hd, Wd
        self.n_masked        = max(1, int(patch_size * patch_size * mask_ratio))
        self.rng             = np.random.default_rng(rng_seed)

    def __len__(self) -> int:
        return self.num_patches

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        P  = self.patch_size
        r0 = self.rng.integers(0, self.Hd - P)
        c0 = self.rng.integers(0, self.Wd - P)

        target    = self.pd_image[:, r0:r0 + P, c0:c0 + P].copy()  # (r², P, P)
        corrupted = target.copy()
        mask      = np.zeros((P, P), dtype=np.float32)   # shared across PD channels

        corrupted, mask = self._apply_masking(target, corrupted, mask)

        return (
            torch.from_numpy(corrupted),              # (r², P, P)
            torch.from_numpy(target),                 # (r², P, P)
            torch.from_numpy(mask).unsqueeze(0),      # (1,  P, P) — broadcasts over r²
        )

    def _apply_masking(
        self,
        target:    np.ndarray,
        corrupted: np.ndarray,
        mask:      np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Replace n_masked positions in-place; return (corrupted, mask)."""
        P   = self.patch_size
        rad = self.neighbor_radius

        flat_idx = self.rng.choice(P * P, size=self.n_masked, replace=False)
        rows, cols = np.unravel_index(flat_idx, (P, P))

        for row, col in zip(rows, cols):
            # Sample a non-zero offset within the neighbour window
            while True:
                dr = int(self.rng.integers(-rad, rad + 1))
                dc = int(self.rng.integers(-rad, rad + 1))
                if dr != 0 or dc != 0:
                    break
            nr = int(np.clip(row + dr, 0, P - 1))
            nc = int(np.clip(col + dc, 0, P - 1))
            corrupted[:, row, col] = target[:, nr, nc]   # all r² channels
            mask[row, col]         = 1.0

        return corrupted, mask


# ============================================================
# 5. Training
# ============================================================

def _numpy_pd(image: np.ndarray, r: int) -> np.ndarray:
    """
    Compute PD transform of a 2-D image in numpy (CPU).

    result[ph*r + pq, row, col] = image_padded[row*r + ph, col*r + pq]

    Equivalent to pd_downsample() but operates directly on numpy arrays
    to avoid GPU round-trips during dataset pre-processing.
    """
    H, W = image.shape
    pad_h = (r - H % r) % r
    pad_w = (r - W % r) % r
    if pad_h > 0 or pad_w > 0:
        image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
    Hp, Wp = image.shape
    Hd, Wd = Hp // r, Wp // r
    # reshape (Hp, Wp) -> (Hd, r, Wd, r) -> transpose (r, r, Hd, Wd) -> (r², Hd, Wd)
    pd = image.reshape(Hd, r, Wd, r).transpose(1, 3, 0, 2).reshape(r * r, Hd, Wd)
    return pd.astype(np.float32)


def train_apbsn(
    model:           nn.Module,
    image:           np.ndarray,
    pd_stride:       int   = 2,
    patch_size:      int   = 64,
    batch_size:      int   = 64,
    num_epochs:      int   = 100,
    learning_rate:   float = 4e-4,
    val_percentage:  float = 0.1,
    device:          torch.device = None,
) -> nn.Module:
    """
    Self-supervised AP-BSN training on a single image.

    The image is PD-transformed once on CPU before dataset creation.
    Loss is MSE computed only at masked pixel positions. The mask shape
    (B, 1, P, P) broadcasts over the r² PD channels (B, r², P, P).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # --- Pre-compute PD domain image (numpy, CPU) ---
    r  = pd_stride
    nc = r * r
    H, W = image.shape
    pd_image = _numpy_pd(image, r)          # (r², Hd, Wd)
    _, Hd, Wd = pd_image.shape

    # Auto-correct patch_size if PD image is smaller
    if patch_size > min(Hd, Wd):
        adjusted = (min(Hd, Wd) // 8) * 8
        print(f"WARNING: patch_size={patch_size} > PD image {Hd}×{Wd}. "
              f"Auto-adjusting to {adjusted}.")
        patch_size = adjusted

    patches_per_epoch = 2000
    n_val   = max(1, int(patches_per_epoch * val_percentage))
    n_train = patches_per_epoch - n_val

    train_ds = APBSNDataset(pd_image, patch_size, n_train, rng_seed=42)
    val_ds   = APBSNDataset(pd_image, patch_size, n_val,   rng_seed=99)

    # num_workers=0 required on Windows to avoid multiprocessing issues
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
    print(f"Device: {device}  |  Parameters: {n_params:,}")
    print(f"PD stride r={r}: {nc} channels  |  "
          f"PD image: {Hd}×{Wd}  (from {H}×{W})")
    print(f"patch_size={patch_size}  batch_size={batch_size}  epochs={num_epochs}")
    print(f"Patches/epoch: train={n_train}  val={n_val}")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        # --- Train ---
        model.train()
        tr_loss, tr_count = 0.0, 0
        for noisy_in, clean_tgt, mask in train_loader:
            noisy_in  = noisy_in.to(device)
            clean_tgt = clean_tgt.to(device)
            mask      = mask.to(device)          # (B, 1, P, P) — broadcast

            optimizer.zero_grad()
            pred = model(noisy_in)
            loss = loss_fn(pred * mask, clean_tgt * mask)
            loss.backward()
            optimizer.step()

            tr_loss  += loss.item()
            tr_count += mask.sum().item() * nc   # each spatial pos → nc loss terms

        # --- Validate ---
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
                  f"{time.time() - t0:.1f}s")

    print("Training complete.")
    return model


# ============================================================
# 6. Inference
# ============================================================

def _run_bsn_on_pd(
    model:  nn.Module,
    pd:     torch.Tensor,   # (1, r², Hd, Wd) on device
    device: torch.device,
) -> torch.Tensor:
    """
    Pass a PD tensor through the BSN.

    The PD-domain spatial dimensions must be divisible by 8 for the U-Net.
    This helper pads to the next multiple of 8 before the forward pass
    and strips the padding from the output.
    """
    _, _, Hd, Wd = pd.shape
    unet_ph = (8 - Hd % 8) % 8
    unet_pw = (8 - Wd % 8) % 8
    if unet_ph > 0 or unet_pw > 0:
        pd = F.pad(pd, (0, unet_pw, 0, unet_ph), mode='reflect')
    with torch.no_grad():
        out = model(pd)
    if unet_ph > 0 or unet_pw > 0:
        out = out[:, :, :Hd, :Wd]    # remove UNet alignment padding
    return out


def predict_apbsn(
    model:      nn.Module,
    image:      np.ndarray,    # (H, W) float32 [0, 1]
    pd_stride:  int,
    device:     torch.device,
    avg_shifts: bool = True,
) -> np.ndarray:
    """
    Run AP-BSN inference on a full image.

    avg_shifts=False (fast):
        image → pd_downsample → BSN → pd_upsample → clip [0, 1]
        One forward pass. May show faint PD-grid texture in flat regions.

    avg_shifts=True (default, AP quality mode):
        For each of the r² spatial shift offsets (p, q) ∈ [0,r)²:
          crop ← image[p:, q:]          # remove first p rows / q cols
          crop → pd_downsample → BSN → pd_upsample → place at [p:, q:]
        Divide accumulator by per-pixel pass count (1 … r²).
        Removes PD-grid artifacts by averaging all alignments.
        r=2 → 4 passes.  r=5 → 25 passes.

    Output is clipped to [0, 1].
    """
    model.eval()
    r = pd_stride
    H, W = image.shape

    if not avg_shifts:
        # --- Single-pass inference ---
        img_t = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)
        pd, pad_hw = pd_downsample(img_t, r)
        pd_out     = _run_bsn_on_pd(model, pd, device)
        denoised   = pd_upsample(pd_out.cpu(), r, pad_hw, (H, W))
        return np.clip(denoised.squeeze().numpy(), 0.0, 1.0).astype(np.float32)

    # --- AP quality mode: average over all r² shift alignments ---
    output_sum = np.zeros((H, W), dtype=np.float64)
    count_sum  = np.zeros((H, W), dtype=np.float64)
    total      = r * r
    pass_idx   = 0

    for p in range(r):
        for q in range(r):
            pass_idx += 1
            crop_np = image[p:, q:]              # (H-p, W-q)
            ch, cw  = crop_np.shape

            crop_t = torch.from_numpy(crop_np.copy()).unsqueeze(0).unsqueeze(0).to(device)
            pd, pad_hw = pd_downsample(crop_t, r)
            pd_out     = _run_bsn_on_pd(model, pd, device)
            result     = pd_upsample(pd_out.cpu(), r, pad_hw, (ch, cw))
            result_np  = result.squeeze().numpy()   # (H-p, W-q)

            # Accumulate at the correct offset position
            ph = result_np.shape[0]   # H-p
            pw = result_np.shape[1]   # W-q
            output_sum[p:p + ph, q:q + pw] += result_np.astype(np.float64)
            count_sum [p:p + ph, q:q + pw] += 1.0

            print(f"  AP inference: {pass_idx}/{total} passes  "
                  f"(shift p={p}, q={q})")

    denoised = (output_sum / np.maximum(count_sum, 1.0)).astype(np.float32)
    return np.clip(denoised, 0.0, 1.0)


# ============================================================
# 7. Save Outputs
# ============================================================

def save_outputs(
    image:    np.ndarray,
    denoised: np.ndarray,
    img_min:  float,
    img_max:  float,
    tif_path: str = "data/denoised_apbsn.tif",
    png_path: str = "data/denoising_apbsn_result.png",
) -> None:
    """Save denoised TIF (original value range) and side-by-side comparison PNG."""
    denoised_orig = (denoised * (img_max - img_min) + img_min).astype(np.float32)
    tifffile.imwrite(tif_path, denoised_orig)
    print(f"Saved: {tif_path}  "
          f"range=[{denoised_orig.min():.3f}, {denoised_orig.max():.3f}]")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image,    cmap='gray')
    axes[0].set_title('Original SEM Image')
    axes[0].axis('off')

    axes[1].imshow(denoised, cmap='gray')
    axes[1].set_title('AP-BSN Denoised')
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
# 8. Main Pipeline
# ============================================================

def main(
    input_path:  str  = "data/test_sem.tif",
    output_path: str  = "data/denoised_sem_apbsn.tif",
    pd_stride:   int  = 2,      # 2 = SEM pixel-indep. noise; 5 = camera sRGB
    patch_size:  int  = 64,     # spatial size in PD domain (= r * patch_size orig. px)
    batch_size:  int  = 64,
    num_epochs:  int  = 100,
    avg_shifts:  bool = True,   # True = AP quality mode (r² passes); False = fast
) -> None:
    """
    Full AP-BSN pipeline: load → train (PD+BSN) → AP-infer → save.

    Parameter quick-reference
    -------------------------
    Scenario               pd_stride  batch_size  epochs  avg_shifts
    SEM pixel-indep.       2          64          100     True  (4 passes)
    Camera sRGB noise      5          32          100     True  (25 passes)
    Fast / preview         2          64          50      False (1 pass)
    Low GPU RAM            2          32          100     True

    Memory note: no tiling is needed during inference because the PD
    downsampling reduces spatial resolution by r.  For a 2048×2048 image
    with r=2, each PD crop is ≤ 1024×1024 with 4 channels (~16 MB); for
    r=5 each crop is ≤ 409×409 with 25 channels (~1.7 MB).  Both fit
    comfortably in an 8 GB GPU.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load image
    image, img_min, img_max = load_sem_image(input_path)
    print(f"Image shape: {image.shape}  "
          f"range: [{img_min:.3f}, {img_max:.3f}]")

    # 2. Build model — in/out channels = pd_stride²
    model = BSNUNet(pd_stride=pd_stride, base_features=32)

    # 3. Train
    model = train_apbsn(
        model, image,
        pd_stride=pd_stride,
        patch_size=patch_size,
        batch_size=batch_size,
        num_epochs=num_epochs,
        device=device,
    )

    # 4. Inference
    mode = (f"AP quality mode ({pd_stride}²={pd_stride**2} passes)"
            if avg_shifts else "single-pass")
    print(f"\nRunning inference [{mode}] ...")
    denoised = predict_apbsn(
        model, image,
        pd_stride=pd_stride,
        device=device,
        avg_shifts=avg_shifts,
    )
    print(f"Denoised range: [{denoised.min():.3f}, {denoised.max():.3f}]")

    # 5. Save
    save_outputs(
        image, denoised, img_min, img_max,
        tif_path=output_path,
        png_path="data/denoising_apbsn_result.png",
    )


if __name__ == '__main__':
    main()

# ============================================================
# SEM Image Denoising — AP-BSN (Lee et al., CVPR 2022)
# Official-style PyTorch port
# ============================================================
# Paper:  "AP-BSN: Self-Supervised Denoising for Real-World Images
#          via Asymmetric PD and Blind-Spot Network"
#          Lee et al., CVPR 2022
# Source: https://github.com/wooseoklee4/AP-BSN
#
# This script ports the official codebase to a self-contained,
# single-file local PyTorch environment without the original
# YAML config / datahandler / trainer framework.
#
# What makes this different from denoise_apbsn_faithful.py
# -------------------------------------------------------
#   faithful.py                  this file (lee.py)
#   -----------                  --------------------
#   PD pre-computed (numpy)   →  PD inside model.forward() (torch)
#   f² sub-images as dataset  →  random crops; model applies PD
#   R3 via full PD pipeline   →  R3 via self.bsn() directly (official)
#   only pd_pad=0             →  pd_pad supported (default 0 for SEM)
#   custom inv_pd loop        →  pixel_shuffle_up_sampling (official)
#
# Architecture overview
# ----------------------
#  APBSN (wrapper)
#    ├── DBSNl (blind-spot network)
#    │     ├── head: 1×1 conv
#    │     ├── DC_branchl(stride=2): CentralMaskedConv + 9 × DCl(dilation=2)
#    │     ├── DC_branchl(stride=3): CentralMaskedConv + 9 × DCl(dilation=3)
#    │     └── tail: 4 × 1×1 convs
#    ├── forward(x, pd=pd_a): pixel_shuffle_down → BSN → pixel_shuffle_up
#    └── denoise(x):          forward(pd=pd_b) → R3 refinement
#
# Training
# --------
#  - Random crop patches from the noisy image
#  - model.forward(crop, pd=pd_a) → L1 vs. noisy crop
#  - No clean reference needed (blind-spot enforces prediction ≠ copy)
#
# Inference
# ---------
#  - model.denoise(full_image): pd_b pass + R3(T=8, p=0.16)
#  - R3: T passes of bsn(mix) where mix = mostly-denoised + few noisy pixels
#
# SEM defaults   : pd_a=2  pd_b=2  pd_pad=0  base_ch=64   epochs=100
# Camera sRGB    : pd_a=5  pd_b=2  pd_pad=2  base_ch=128  epochs=100
#
# Usage
# -----
#   python test_sem.py                          # generate test image if needed
#   python denoise_apbsn_lee.py                 # SEM mode (defaults)
#   python denoise_apbsn_lee.py --pd_a 5 --pd_b 2 --pd_pad 2 --base_ch 128
#   python denoise_apbsn_lee.py --no_r3         # single-pass inference
#   python denoise_apbsn_lee.py --save_model checkpoints/apbsn_lee.pth
#   python denoise_apbsn_lee.py --load_model checkpoints/apbsn_lee.pth
# ============================================================
# Requirements: torch>=2.0  tifffile  matplotlib  numpy
# ============================================================

import argparse
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import time
from typing import Tuple, Optional

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
    """Load SEM image, normalize to float32 [0, 1] grayscale."""
    img = tifffile.imread(path).astype(np.float32)
    if img.ndim == 3 and img.shape[-1] in (3, 4):
        img = img[..., :3] @ np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
    img_min, img_max = float(img.min()), float(img.max())
    img = (img - img_min) / (img_max - img_min + 1e-8)
    return img, img_min, img_max


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

    Steps:
      1. pixel_unshuffle(f): group pixels by phase offset into channels
         (B, C, H, W) → (B, C·f², H//f, W//f)
      2. optional reflection-free pad in the sub-sampled domain
      3. view / permute / reshape: interleave sub-sampled positions back
         into a spatial image of size (H + 2·f·pad, W + 2·f·pad)

    Requires H and W to be divisible by f (use pad_to_multiple before call
    if needed, or choose crop_size divisible by pd_a during training).
    """
    b, c, H, W = x.shape
    unshuffled = F.pixel_unshuffle(x, f)  # (B, C·f², H//f, W//f)
    if pad != 0:
        unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
    Hp = H // f + 2 * pad
    Wp = W // f + 2 * pad
    # split channel dim into (C, f_h, f_w) then interleave with spatial dims
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
    b, c, H, W = x.shape   # H = H_orig + 2·f·pad
    # undo the interleaved spatial arrangement
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
    Conv2d with the center kernel weight permanently zeroed (blind-spot).

    The output at (i, j) cannot depend on the input at (i, j) — the
    network is structurally blind to the pixel it is predicting.
    Mask is stored as a persistent buffer so it moves with .to(device).

    Kernel size per branch: 2·stride − 1
      stride=2 → 3×3 masked conv
      stride=3 → 5×5 masked conv
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

    Sequence:
      CentralMaskedConv2d(k=2·stride−1) → 3 × (1×1+ReLU) →
      num_module × DCl(dilation=stride) → 1×1 + ReLU
    """
    def __init__(self, stride: int, base_ch: int, num_module: int):
        super().__init__()
        kernel  = 2 * stride - 1
        padding = kernel // 2
        # Official layer order (DBSNl.py):
        #   CentralMaskedConv → ReLU → [1×1 Conv + ReLU] × 2 → DCl × N → 1×1 Conv + ReLU
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

    head  : 1×1 conv (channel projection)
    branch1: DC_branchl(stride=2)
    branch2: DC_branchl(stride=3)
    tail  : concat → four 1×1 projections → output

    Paper defaults: base_ch=128, num_module=9.
    This script defaults to base_ch=64 for lighter SEM training.
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

    forward(img, pd=pd_a)
    ---------------------
    Training path. Applies pixel_shuffle_down_sampling with pd (= pd_a),
    runs DBSNl, then pixel_shuffle_up_sampling back.  The crop must have
    spatial dims divisible by pd.

    denoise(x)
    ----------
    Inference path (official logic).
      1. Pad x to be divisible by pd_b.
      2. First pass: img_pd_bsn = forward(x, pd=pd_b).
      3. R3 (T passes):
           mask ~ Bernoulli(R3_p)
           tmp = img_pd_bsn;  tmp[mask] = x[mask]   # re-inject noisy
           pad(pd_pad, reflect) → bsn(tmp) → crop    # BSN directly, no PD
      4. Average all R3_T outputs.
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
        # pad to nearest multiple of pd_b
        pad_h = (self.pd_b - h % self.pd_b) % self.pd_b
        pad_w = (self.pd_b - w % self.pd_b) % self.pd_b
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        # --- first pass: PD(pd_b) → BSN → PD⁻¹ ---
        img_pd_bsn = self.forward(x, pd=self.pd_b)

        if not self.R3:
            return img_pd_bsn[:, :, :h, :w]

        # --- R3: T passes of bsn( mostly-denoised + few noisy ) ---
        # denoised[..., t] collects each refinement result
        denoised = torch.empty(*x.shape, self.R3_T, device=x.device)
        for t in range(self.R3_T):
            # random mask: True where we re-inject original noisy pixels
            mask = torch.rand_like(x) < self.R3_p
            tmp  = img_pd_bsn.clone()
            tmp[mask] = x[mask]                                     # re-inject noise

            # BSN directly (no PD), with optional reflection padding
            p = self.pd_pad
            if p > 0:
                tmp = F.pad(tmp, (p, p, p, p), mode='reflect')
                denoised[..., t] = self.bsn(tmp)[:, :, p:-p, p:-p]
            else:
                denoised[..., t] = self.bsn(tmp)

        return torch.mean(denoised, dim=-1)[:, :, :h, :w]


# ============================================================
# 5. Training Dataset
# ============================================================

class PatchDataset(Dataset):
    """
    Random patch dataset for APBSN training.

    Crops are extracted from the raw noisy image (no PD preprocessing).
    PD is applied inside APBSN.forward() during training.

    crop_size must be divisible by pd_a for pixel_shuffle to work.
    """
    def __init__(
        self,
        image:       np.ndarray,     # (H, W) float32 [0, 1]
        crop_size:   int   = 64,
        num_patches: int   = 2000,
        rng_seed:    Optional[int] = None,
    ):
        H, W = image.shape
        assert crop_size <= H and crop_size <= W, (
            f"crop_size={crop_size} exceeds image size {H}×{W}"
        )
        self.image       = image
        self.H, self.W   = H, W
        self.crop_size   = crop_size
        self.num_patches = num_patches
        self.rng         = np.random.default_rng(rng_seed)

    def __len__(self) -> int:
        return self.num_patches

    def __getitem__(self, _: int) -> torch.Tensor:
        P  = self.crop_size
        r0 = self.rng.integers(0, self.H - P + 1)
        c0 = self.rng.integers(0, self.W - P + 1)
        patch = self.image[r0:r0 + P, c0:c0 + P][np.newaxis].copy()  # (1, P, P)
        return torch.from_numpy(patch)


# ============================================================
# 6. Training
# ============================================================

def train_apbsn(
    model:         APBSN,
    image:         np.ndarray,
    crop_size:     int   = 64,
    batch_size:    int   = 32,
    num_epochs:    int   = 100,
    learning_rate: float = 1e-4,
    device:        Optional[torch.device] = None,
) -> APBSN:
    """
    Train AP-BSN on a single noisy image.

    Loss (official: recon_self.py → self_L1):
        L = ‖APBSN.forward(crop, pd=pd_a) − crop‖₁

    The blind-spot (CentralMaskedConv2d) prevents trivial copying, so
    the L1 loss against the noisy crop is a valid self-supervised objective.

    Optimizer and schedule match the official SIDD config:
      Adam (lr=1e-4, betas=(0.9, 0.999)), StepLR (γ=0.1 every step_epochs).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    H, W = image.shape

    # crop_size must be divisible by pd_a
    pd_a = model.pd_a
    if crop_size % pd_a != 0:
        crop_size = (crop_size // pd_a) * pd_a
        print(f"INFO: crop_size rounded down to {crop_size} (must be divisible by pd_a={pd_a}).")
    if crop_size > H or crop_size > W:
        crop_size = min(H, W)
        crop_size = (crop_size // pd_a) * pd_a
        print(f"INFO: crop_size auto-adjusted to {crop_size}.")

    n_train, n_val = 1800, 200
    train_ds = PatchDataset(image, crop_size, n_train, rng_seed=42)
    val_ds   = PatchDataset(image, crop_size, n_val,   rng_seed=99)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0)

    optimizer  = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    step_epochs = max(1, num_epochs // 3)          # decay LR twice over training
    scheduler  = optim.lr_scheduler.StepLR(optimizer, step_size=step_epochs, gamma=0.1)
    loss_fn    = nn.L1Loss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}  |  Parameters: {n_params:,}")
    print(f"Asymmetric PD — train pd_a={pd_a}  infer pd_b={model.pd_b}  pd_pad={model.pd_pad}")
    print(f"crop_size={crop_size}  batch_size={batch_size}  epochs={num_epochs}")
    print(f"R3 at inference: {'enabled' if model.R3 else 'disabled'} "
          f"(T={model.R3_T}, p={model.R3_p})")

    pix_train = n_train * crop_size * crop_size
    pix_val   = n_val   * crop_size * crop_size

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        # --- Train ---
        model.train()
        tr_loss = 0.0
        for batch in train_loader:
            batch  = batch.to(device)
            output = model(batch)                  # APBSN.forward() with pd=pd_a
            loss   = loss_fn(output, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * batch.numel()

        # --- Validate ---
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
# 7. Inference
# ============================================================

def predict(
    model:  APBSN,
    image:  np.ndarray,     # (H, W) float32 [0, 1]
    device: torch.device,
) -> np.ndarray:
    """
    Run full AP-BSN inference (APBSN.denoise).

    The image is converted to a (1, 1, H, W) tensor, passed through
    APBSN.denoise() which handles PD(pd_b) + R3, then converted back.
    For very large images consider reducing R3_T or disabling R3 if OOM.
    """
    model.eval()
    x = torch.from_numpy(image[np.newaxis, np.newaxis]).to(device)  # (1,1,H,W)

    if model.R3:
        print(f"Inference: pd_b={model.pd_b}  + R3(T={model.R3_T}, p={model.R3_p})")
    else:
        print(f"Inference: pd_b={model.pd_b}  (R3 disabled)")

    with torch.no_grad():
        out = model.denoise(x)

    return np.clip(out.cpu().squeeze().numpy(), 0.0, 1.0).astype(np.float32)


# ============================================================
# 8. Save Outputs
# ============================================================

def save_outputs(
    image:    np.ndarray,
    denoised: np.ndarray,
    img_min:  float,
    img_max:  float,
    tif_path: str = "data/denoised_sem_apbsn_lee.tif",
    png_path: str = "data/denoising_result_APBSN_lee.png",
) -> None:
    """Save denoised TIF (original value range) and side-by-side PNG."""
    os.makedirs(os.path.dirname(tif_path) or ".", exist_ok=True)
    denoised_orig = (denoised * (img_max - img_min) + img_min).astype(np.float32)
    tifffile.imwrite(tif_path, denoised_orig)
    print(f"Saved: {tif_path}  "
          f"range=[{denoised_orig.min():.3f}, {denoised_orig.max():.3f}]")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image,    cmap='gray'); axes[0].set_title('Original SEM');    axes[0].axis('off')
    axes[1].imshow(denoised, cmap='gray'); axes[1].set_title('AP-BSN (Lee)');    axes[1].axis('off')
    diff = np.abs(image - denoised) * 3
    axes[2].imshow(diff,     cmap='hot');  axes[2].set_title('Difference (×3)'); axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.show()
    print(f"Saved: {png_path}")


# ============================================================
# 9. Main Pipeline
# ============================================================

def main() -> None:
    """
    Full AP-BSN (Lee et al. style) pipeline:
        load → build APBSN → train(pd_a, L1) → denoise(pd_b, R3) → save

    Parameter quick-reference
    -------------------------
    Scenario               pd_a  pd_b  pd_pad  base_ch  epochs
    SEM pixel-independent  2     2     0       64       100     (default)
    Camera sRGB noise      5     2     2       128      100
    Fast / preview         2     2     0       32       50
    Low GPU RAM            2     2     0       32       100
    """
    parser = argparse.ArgumentParser(
        description="AP-BSN (Lee et al., CVPR 2022) — official-style PyTorch port"
    )
    parser.add_argument('--input',      type=str,   default='data/test_sem.tif',
                        help='Input .tif/.tiff/.png path')
    parser.add_argument('--output',     type=str,   default='',
                        help='Output .tif path (default: data/denoised_sem_apbsn_lee.tif)')
    # PD parameters
    parser.add_argument('--pd_a',       type=int,   default=2,
                        help='PD stride for TRAINING  (paper: 5 for camera sRGB; 2 for SEM)')
    parser.add_argument('--pd_b',       type=int,   default=2,
                        help='PD stride for INFERENCE (paper: 2)')
    parser.add_argument('--pd_pad',     type=int,   default=0,
                        help='PD padding (paper: 2 for camera; 0 for SEM)')
    # DBSNl parameters
    parser.add_argument('--base_ch',    type=int,   default=64,
                        help='DBSNl base channels (paper: 128; SEM default: 64)')
    parser.add_argument('--num_module', type=int,   default=9,
                        help='DCl residual blocks per branch (paper: 9)')
    # Training parameters
    parser.add_argument('--crop_size',  type=int,   default=64,
                        help='Training crop size (must be divisible by pd_a)')
    parser.add_argument('--batch_size', type=int,   default=32)
    parser.add_argument('--epochs',     type=int,   default=100)
    parser.add_argument('--lr',         type=float, default=1e-4,
                        help='Initial learning rate (paper: 1e-4)')
    # R3 parameters
    parser.add_argument('--R3_T',       type=int,   default=8,
                        help='R3 refinement passes (paper: 8)')
    parser.add_argument('--R3_p',       type=float, default=0.16,
                        help='R3 random replacement probability (paper: 0.16)')
    parser.add_argument('--no_r3',      action='store_true',
                        help='Disable R3 (faster single-pass inference)')
    # Model persistence
    parser.add_argument('--save_model', type=str,   default='',
                        help='Save trained model weights to this path')
    parser.add_argument('--load_model', type=str,   default='',
                        help='Load weights from this path (skip training)')
    # Hardware
    parser.add_argument('--device',     type=str,   default=None,
                        help='Device override: cuda, cpu, cuda:1 … (default: auto)')
    args = parser.parse_args()

    input_path  = args.input
    output_path = args.output or 'data/denoised_sem_apbsn_lee.tif'
    device      = torch.device(
        args.device if args.device
        else ('cuda' if torch.cuda.is_available() else 'cpu')
    )

    # 1. Load image
    image, img_min, img_max = load_sem_image(input_path)
    print(f"Image: {image.shape}  range=[{img_min:.3f}, {img_max:.3f}]")

    # 2. Build APBSN (official wrapper: DBSNl + PD + R3)
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
        print(f"Symmetric PD mode (pd_a = pd_b = {args.pd_a}): suitable for SEM pixel-independent noise.")
    else:
        print(f"Asymmetric PD mode (pd_a={args.pd_a} train, pd_b={args.pd_b} infer): "
              f"suited for spatially-correlated camera noise.")

    # 3. Train or load
    if args.load_model:
        ckpt = torch.load(args.load_model, map_location=device)
        model.load_state_dict(ckpt)
        model = model.to(device)
        print(f"Loaded model from {args.load_model} — skipping training.")
    else:
        model = train_apbsn(
            model, image,
            crop_size     = args.crop_size,
            batch_size    = args.batch_size,
            num_epochs    = args.epochs,
            learning_rate = args.lr,
            device        = device,
        )
        if args.save_model:
            os.makedirs(os.path.dirname(args.save_model) or ".", exist_ok=True)
            torch.save(model.state_dict(), args.save_model)
            print(f"Model saved to {args.save_model}")

    # 4. Inference: APBSN.denoise(pd_b) + optional R3
    print(f"\nRunning inference ...")
    denoised = predict(model, image, device)
    print(f"Denoised range: [{denoised.min():.3f}, {denoised.max():.3f}]")

    # 5. Save
    save_outputs(
        image, denoised, img_min, img_max,
        tif_path = output_path,
        png_path = 'data/denoising_result_APBSN_lee.png',
    )


if __name__ == '__main__':
    main()

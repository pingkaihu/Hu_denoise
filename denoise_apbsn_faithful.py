# ============================================================
# SEM Image Denoising — AP-BSN (Paper-Faithful Implementation)
# ============================================================
# Paper: "AP-BSN: Self-Supervised Denoising for Real-World Images
#         via Asymmetric PD and Blind-Spot Network"
#         Lee et al., CVPR 2022
# arXiv:  https://arxiv.org/abs/2203.11799
# GitHub: https://github.com/wooseoklee4/AP-BSN
#
# Three paper contributions faithfully implemented:
#
#   1. DBSNl architecture
#        - Dilated conv branches (dilation=2 and dilation=3)
#        - CentralMaskedConv2d: ARCHITECTURAL blind-spot (not training-time masking)
#        - kernel_size = 2*stride-1 per branch (3 for stride=2, 5 for stride=3)
#        - L1 loss on ALL pixels (no selective masked-pixel loss)
#
#   2. Asymmetric PD (the "A" in AP-BSN)
#        - Training   uses pd_a (large, e.g. 5): decorrelates spatially-correlated noise
#        - Inference  uses pd_b (small, e.g. 2): preserves image structure
#        - Same model (grayscale, 1-channel in/out) used at both strides
#        - For SEM (pixel-independent noise): pd_a = pd_b = 2 is sufficient
#
#   3. R3 — Random-Replacing Refinement (post-processing, no extra parameters)
#        - T=8 passes: each pass randomly replaces R3_p=0.16 fraction of
#          the denoised pixels with original noisy pixels, runs BSN again
#        - Average all T outputs for the final denoised image
#
# Comparison with denoise_apbsn.py (what changed):
#   BEFORE                         AFTER (this file)
#   -------                        ------
#   BSNUNet (standard U-Net)    →  DBSNl (dilated blind-spot network)
#   N2V masking at train-time   →  CentralMaskedConv2d (architectural)
#   MSE on masked pixels only   →  L1 on ALL pixels
#   Single pd_stride            →  pd_a (train) ≠ pd_b (infer)
#   avg_shifts workaround       →  R3 refinement (from paper §3.3)
#
# SEM recommended defaults:
#   pd_a=2  pd_b=2  base_ch=64  num_module=9  R3_T=8  R3_p=0.16
#
# Camera sRGB (spatially-correlated ISP noise):
#   pd_a=5  pd_b=2  base_ch=128  num_module=9  R3_T=8  R3_p=0.16
#
# Usage:
#   python test_sem.py                            # generate test image if needed
#   python denoise_apbsn_faithful.py              # SEM mode (pd_a=2, pd_b=2)
#   python denoise_apbsn_faithful.py --pd_a 5 --pd_b 2 --base_ch 128  # camera sRGB
#   python denoise_apbsn_faithful.py --no_r3      # skip R3 (faster preview)
# ============================================================
# Requirements: torch>=2.0  tifffile  matplotlib  numpy
# ============================================================

import argparse
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

torch.set_float32_matmul_precision('high')


# ============================================================
# 1. Image Loading
# ============================================================

def load_sem_image(path: str) -> Tuple[np.ndarray, float, float]:
    """Load SEM image, normalize to float32 [0, 1] grayscale."""
    img = tifffile.imread(path).astype(np.float32)
    if img.ndim == 3 and img.shape[-1] == 3:
        img = img @ np.array([0.2989, 0.5870, 0.1140])
    img_min, img_max = float(img.min()), float(img.max())
    img = (img - img_min) / (img_max - img_min + 1e-8)
    return img, img_min, img_max


# ============================================================
# 2. PD Operations (numpy, CPU)
# ============================================================

def _numpy_pd(image: np.ndarray, r: int) -> np.ndarray:
    """
    Pixel-Shuffle Downsampling: (H, W) → (r², Hd, Wd).

    Reflect-pads to the nearest multiple of r, then rearranges pixels
    by their (row % r, col % r) phase offset. Each of the r² output
    channels is a spatially-subsampled version of the input:

        output[ph*r + pq, row, col] = image_padded[row*r + ph, col*r + pq]

    With pd_a large (e.g. 5), spatially-correlated camera noise becomes
    approximately pixel-independent in the PD domain, satisfying the
    BSN training assumption.
    """
    H, W   = image.shape
    pad_h  = (r - H % r) % r
    pad_w  = (r - W % r) % r
    if pad_h > 0 or pad_w > 0:
        image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
    Hp, Wp = image.shape
    Hd, Wd = Hp // r, Wp // r
    # (Hp, Wp) → (Hd, r, Wd, r) → transpose (r, r, Hd, Wd) → (r², Hd, Wd)
    pd = image.reshape(Hd, r, Wd, r).transpose(1, 3, 0, 2).reshape(r * r, Hd, Wd)
    return pd.astype(np.float32)


def _numpy_pd_inv(pd: np.ndarray, r: int, orig_H: int, orig_W: int) -> np.ndarray:
    """
    Inverse PD: (r², Hd, Wd) → (orig_H, orig_W).

    Reverses _numpy_pd exactly. The reflect-padding added during the
    forward pass is removed by cropping to (orig_H, orig_W).
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
    Convolutional layer with the center kernel weight permanently zeroed.

    This is the architectural blind-spot mechanism used in DBSNl (paper §3.1).
    The output at position (i, j) cannot depend on the input at (i, j),
    regardless of whether the network is in train or eval mode.

    Unlike N2V-style training-time masking, this guarantee holds at inference
    and requires no special handling — the network is structurally blind.

    Kernel size is chosen as 2*branch_stride - 1:
      Branch stride=2 → kernel_size=3 (3×3 masked conv)
      Branch stride=3 → kernel_size=5 (5×5 masked conv)

    The mask is stored as a persistent buffer so it moves with .to(device).
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
    """
    Dilated residual block (DCl) from DBSNl (paper Fig. 2).

    Structure: dilated 3×3 conv → ReLU → 1×1 conv → + residual.

    The dilation rate equals the branch stride parameter (2 or 3).
    No BatchNorm: the paper's DBSNl omits normalisation layers.
    """
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

    Sequence per branch (stride ∈ {2, 3}):
      CentralMaskedConv2d(kernel=2*stride-1)   ← blind-spot entry point
      → 1×1 Conv + ReLU  ×3                    ← channel mixing
      → DCl(dilation=stride) × num_module       ← dilated residual blocks
      → 1×1 Conv + ReLU                         ← exit mixing

    The CentralMaskedConv2d is the ONLY place where the blind-spot is
    enforced.  All subsequent operations freely access any feature
    position — but because the feature at the center was computed without
    the center input pixel, the property propagates automatically.
    """
    def __init__(self, stride: int, base_ch: int, num_module: int):
        super().__init__()
        kernel = 2 * stride - 1     # 3 for stride=2, 5 for stride=3
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

    Architecture:
      Head    : 1×1 Conv + ReLU  (channel expansion, no blind-spot here)
      Branch1 : DC_branchl(stride=2)  (blind-spot via CentralMaskedConv2d)
      Branch2 : DC_branchl(stride=3)  (blind-spot via CentralMaskedConv2d)
      Tail    : concat → 1×1×4 fusions → output

    Paper defaults: base_ch=128, num_module=9.
    This file defaults to base_ch=64 for faster SEM training; use
    base_ch=128 to match paper performance on real camera images.

    The network operates on single-channel (grayscale) images.
    The PD operation is applied externally before passing to this model.
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
            nn.Conv2d(base_ch * 2, base_ch,      kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch,     base_ch // 2, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch // 2, base_ch // 2, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch // 2, in_ch,        kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h  = self.head(x)
        b1 = self.branch1(h)
        b2 = self.branch2(h)
        return self.tail(torch.cat([b1, b2], dim=1))


# ============================================================
# 4. Dataset (Asymmetric PD: training uses pd_a)
# ============================================================

class APBSNDataset(Dataset):
    """
    Paper-faithful training dataset.

    Applies PD with the TRAINING stride pd_a to produce pd_a² sub-images.
    Each sub-image is approximately pixel-independent (guaranteed when pd_a
    exceeds the noise correlation length). Training samples are single-channel
    patches drawn uniformly from all sub-images.

    Loss target = noisy patch itself (Noise2Noise / BSN principle):
        L = ‖DBSNl(patch) − patch‖₁

    No N2V-style masking is needed because the CentralMaskedConv2d in
    DBSNl provides the blind-spot at the architectural level.

    Parameters
    ----------
    image       : (H, W) float32 [0, 1] noisy SEM image
    pd_a        : PD stride for training  (paper: 5 for camera; 2 for SEM)
    patch_size  : spatial patch size in the PD domain
    num_patches : virtual epoch length
    rng_seed    : numpy seed for reproducibility
    """
    def __init__(
        self,
        image:       np.ndarray,
        pd_a:        int = 2,
        patch_size:  int = 64,
        num_patches: int = 2000,
        rng_seed:    int = None,
    ):
        subs = _numpy_pd(image, pd_a)     # (pd_a², Hd, Wd)
        self.subs  = subs
        self.n_s, self.Hd, self.Wd = subs.shape
        self.P           = patch_size
        self.num_patches = num_patches
        self.rng         = np.random.default_rng(rng_seed)

        assert patch_size % 8 == 0, \
            f"patch_size must be divisible by 8 (convention), got {patch_size}"
        if patch_size > min(self.Hd, self.Wd):
            raise ValueError(
                f"patch_size={patch_size} exceeds PD-domain size {self.Hd}×{self.Wd} "
                f"(image {image.shape}, pd_a={pd_a}). "
                f"Use --patch_size ≤ {(min(self.Hd, self.Wd) // 8) * 8}."
            )

    def __len__(self) -> int:
        return self.num_patches

    def __getitem__(self, _idx: int) -> torch.Tensor:
        P  = self.P
        si = self.rng.integers(0, self.n_s)
        r0 = self.rng.integers(0, self.Hd - P + 1)
        c0 = self.rng.integers(0, self.Wd - P + 1)
        patch = self.subs[si, r0:r0 + P, c0:c0 + P][np.newaxis].copy()  # (1, P, P)
        return torch.from_numpy(patch)


# ============================================================
# 5. Training
# ============================================================

def train_apbsn(
    model:         nn.Module,
    image:         np.ndarray,
    pd_a:          int   = 2,
    patch_size:    int   = 64,
    batch_size:    int   = 32,
    num_epochs:    int   = 100,
    learning_rate: float = 4e-4,
    device:        torch.device = None,
) -> nn.Module:
    """
    Train DBSNl on PD-domain patches (Asymmetric PD training stride pd_a).

    Loss (paper Eq. 2):
        L_BSN = ‖I^s_BSN − I_N‖₁

    where I_N is the noisy patch and I^s_BSN = DBSNl(I_N). All pixels
    contribute to the loss (no selective masked-pixel weighting).

    The blind-spot is enforced by CentralMaskedConv2d, so the model
    cannot trivially copy the input — it must predict each center pixel
    from its spatial context.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Check PD-domain size and auto-adjust patch_size if needed
    pd_image = _numpy_pd(image, pd_a)
    _, Hd, Wd = pd_image.shape
    if patch_size > min(Hd, Wd):
        patch_size = min(Hd, Wd)
        print(f"WARNING: patch_size auto-adjusted to {patch_size} "
              f"(PD domain is {Hd}×{Wd} with pd_a={pd_a}).")

    n_train, n_val = 1800, 200
    train_ds = APBSNDataset(image, pd_a, patch_size, n_train, rng_seed=42)
    val_ds   = APBSNDataset(image, pd_a, patch_size, n_val,   rng_seed=99)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )
    loss_fn = nn.L1Loss()   # Paper uses L1 loss (not MSE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}  |  Parameters: {n_params:,}")
    print(f"Asymmetric PD — training stride pd_a={pd_a} "
          f"→ {pd_a**2} sub-images each {Hd}×{Wd}")
    print(f"patch_size={patch_size}  batch_size={batch_size}  epochs={num_epochs}")

    pix_train = n_train * patch_size * patch_size
    pix_val   = n_val   * patch_size * patch_size

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        # --- Train ---
        model.train()
        tr_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            pred  = model(batch)
            loss  = loss_fn(pred, batch)   # L1(output, noisy_patch)
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
                  f"{time.time() - t0:.1f}s")

    print("Training complete.")
    return model


# ============================================================
# 6. Single-Pass Inference (with pd_b)
# ============================================================

def _infer_single_pass(
    model:  nn.Module,
    image:  np.ndarray,   # (H, W) float32 [0, 1]
    pd_b:   int,
    device: torch.device,
) -> np.ndarray:
    """
    One inference pass with PD stride pd_b.

    Steps (paper §3.2):
      1. PD(pd_b): image → pd_b² sub-images of shape (Hd, Wd)
      2. BSN: denoise each sub-image independently (1-ch in, 1-ch out)
      3. PD⁻¹: reassemble the denoised sub-images to (H, W)

    Using pd_b < pd_a at inference (paper: 2 vs 5) preserves fine image
    structure that would otherwise be degraded by heavy downsampling.
    The asymmetry is the core contribution of AP-BSN.
    """
    H, W = image.shape
    subs = _numpy_pd(image, pd_b)   # (pd_b², Hd, Wd)
    n_s  = pd_b * pd_b

    # Stack all pd_b² sub-images as a batch: (n_s, 1, Hd, Wd)
    batch_t = torch.from_numpy(subs[:, np.newaxis]).to(device)
    model.eval()
    with torch.no_grad():
        out_t = model(batch_t)     # (n_s, 1, Hd, Wd)
    out_subs = out_t.cpu().squeeze(1).numpy()   # (pd_b², Hd, Wd)

    return np.clip(_numpy_pd_inv(out_subs, pd_b, H, W), 0.0, 1.0).astype(np.float32)


# ============================================================
# 7. R3 — Random-Replacing Refinement (paper §3.3)
# ============================================================

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
    Full AP-BSN inference with R3 random-replacing refinement.

    R3 algorithm (paper §3.3, official code: R3_T=8, R3_p=0.16):
      D₀ = BSN_infer(image)
      for t = 1 … R3_T−1:
          M  ~ Bernoulli(R3_p)           [random binary mask]
          x_mix = (1−M)·D₀ + M·image    [mix denoised + noisy]
          D_t = BSN_infer(x_mix)
      output = mean(D₀, D₁, …, D_{T−1})

    By randomly reintroducing noise at some positions we obtain T
    diverse denoised candidates; their average further suppresses
    noise without any additional learnable parameters.

    Set use_r3=False for a fast single-pass result (lower quality).
    """
    D0 = _infer_single_pass(model, image, pd_b, device)
    if not use_r3:
        return D0

    results = [D0]
    rng = np.random.default_rng(seed=0)
    for t in range(1, R3_T):
        mask  = (rng.random(image.shape) < R3_p).astype(np.float32)
        mixed = (1.0 - mask) * D0 + mask * image   # replace R3_p fraction with noisy
        Dt    = _infer_single_pass(model, mixed, pd_b, device)
        results.append(Dt)
        print(f"  R3: pass {t}/{R3_T - 1}")

    return np.clip(np.mean(results, axis=0), 0.0, 1.0).astype(np.float32)


# ============================================================
# 8. Save Outputs
# ============================================================

def save_outputs(
    image:    np.ndarray,
    denoised: np.ndarray,
    img_min:  float,
    img_max:  float,
    tif_path: str = "data/denoised_sem_apbsn_faithful.tif",
    png_path: str = "data/denoising_result_APBSN_faithful.png",
) -> None:
    """Save denoised TIF (original value range) and side-by-side PNG."""
    os.makedirs(os.path.dirname(tif_path) or ".", exist_ok=True)
    denoised_orig = (denoised * (img_max - img_min) + img_min).astype(np.float32)
    tifffile.imwrite(tif_path, denoised_orig)
    print(f"Saved: {tif_path}  "
          f"range=[{denoised_orig.min():.3f}, {denoised_orig.max():.3f}]")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image,    cmap='gray'); axes[0].set_title('Original SEM');       axes[0].axis('off')
    axes[1].imshow(denoised, cmap='gray'); axes[1].set_title('AP-BSN (Faithful)');  axes[1].axis('off')
    diff = np.abs(image - denoised) * 3
    axes[2].imshow(diff,     cmap='hot');  axes[2].set_title('Difference (×3)');    axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.show()
    print(f"Saved: {png_path}")


# ============================================================
# 9. Main Pipeline
# ============================================================

def main() -> None:
    """
    Full AP-BSN (faithful) pipeline:
        load → build DBSNl → train(pd_a, L1) → infer(pd_b) → R3 → save

    Parameter quick-reference
    -------------------------
    Scenario               pd_a  pd_b  base_ch  epochs  R3
    SEM pixel-independent  2     2     64       100     True  (default)
    Camera sRGB noise      5     2     128      100     True
    Fast / preview         2     2     32       50      False
    Low GPU RAM            2     2     32       100     True
    """
    parser = argparse.ArgumentParser(
        description="AP-BSN faithful: DBSNl + Asymmetric PD + R3 (CVPR 2022)"
    )
    parser.add_argument('--input',      type=str,   default='data/test_sem.tif',
                        help='Input .tif/.tiff/.png path')
    parser.add_argument('--output',     type=str,   default='',
                        help='Output .tif path (default: data/denoised_sem_apbsn_faithful.tif)')
    parser.add_argument('--pd_a',       type=int,   default=2,
                        help='PD stride for TRAINING  (paper: 5 for camera sRGB; 2 for SEM)')
    parser.add_argument('--pd_b',       type=int,   default=2,
                        help='PD stride for INFERENCE (paper: 2)')
    parser.add_argument('--base_ch',    type=int,   default=64,
                        help='DBSNl base channels per branch (paper: 128)')
    parser.add_argument('--num_module', type=int,   default=9,
                        help='DCl residual blocks per branch (paper: 9)')
    parser.add_argument('--patch_size', type=int,   default=64,
                        help='Training patch size in the PD domain')
    parser.add_argument('--batch_size', type=int,   default=32)
    parser.add_argument('--epochs',     type=int,   default=100)
    parser.add_argument('--R3_T',       type=int,   default=8,
                        help='R3 refinement passes (paper: 8)')
    parser.add_argument('--R3_p',       type=float, default=0.16,
                        help='R3 random replacement probability (paper: 0.16)')
    parser.add_argument('--no_r3',      action='store_true',
                        help='Disable R3 (faster single-pass inference)')
    parser.add_argument('--device',     type=str,   default=None,
                        help='Device override: cuda, cpu, cuda:1 … (default: auto)')
    args = parser.parse_args()

    input_path  = args.input
    output_path = args.output or 'data/denoised_sem_apbsn_faithful.tif'
    device      = torch.device(
        args.device if args.device
        else ('cuda' if torch.cuda.is_available() else 'cpu')
    )

    # 1. Load
    image, img_min, img_max = load_sem_image(input_path)
    print(f"Image: {image.shape}  range=[{img_min:.3f}, {img_max:.3f}]")
    print(f"Asymmetric PD: pd_a={args.pd_a} (train)  pd_b={args.pd_b} (infer)")
    if args.pd_a == args.pd_b:
        print("  Note: pd_a == pd_b — symmetric mode (appropriate for SEM pixel-independent noise)")
    else:
        print("  Note: pd_a > pd_b — asymmetric mode (breaks spatial correlation for camera noise)")

    # 2. Build model
    model = DBSNl(in_ch=1, base_ch=args.base_ch, num_module=args.num_module)

    # 3. Train with pd_a, L1 loss on all pixels
    model = train_apbsn(
        model, image,
        pd_a        = args.pd_a,
        patch_size  = args.patch_size,
        batch_size  = args.batch_size,
        num_epochs  = args.epochs,
        device      = device,
    )

    # 4. Inference with pd_b + optional R3 refinement
    use_r3 = not args.no_r3
    mode_str = f"pd_b={args.pd_b}"
    if use_r3:
        mode_str += f" + R3(T={args.R3_T}, p={args.R3_p})"
    print(f"\nRunning inference [{mode_str}] ...")

    denoised = predict_apbsn(
        model, image,
        pd_b   = args.pd_b,
        device = device,
        R3_T   = args.R3_T,
        R3_p   = args.R3_p,
        use_r3 = use_r3,
    )
    print(f"Denoised range: [{denoised.min():.3f}, {denoised.max():.3f}]")

    # 5. Save
    save_outputs(
        image, denoised, img_min, img_max,
        tif_path = output_path,
        png_path = 'data/denoising_result_APBSN_faithful.png',
    )


if __name__ == '__main__':
    main()

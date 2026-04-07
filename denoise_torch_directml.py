# ============================================================
# SEM Image Denoising — Noise2Void (PyTorch + Intel DirectML)
# ============================================================
# Requirements: torch  tifffile  matplotlib  numpy  torch-directml
#   pip install torch-directml
# Usage:
#   python test_sem.py                   # generate synthetic test image
#   python denoise_torch_directml.py     # train + denoise -> denoised_sem_directml.tif
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
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

torch.set_float32_matmul_precision('high')


# ============================================================
# Device Selection — DirectML > CPU fallback
#
# 選項一（DirectML）效益不佳時的替代方案：
#
#   選項三：純 CPU + Intel MKL 最佳化
#   ─────────────────────────────────
#   適用情境：
#     - Intel iGPU 為舊款（UHD 620/630 等），DirectML 加速效益低於 CPU
#     - torch_directml 安裝失敗或出現算子不支援的錯誤
#     - 模型較小（base_features <= 16）時 CPU 與 iGPU 差距不明顯
#
#   使用方式：直接將 main() 中的 device 改為：
#
#       import torch
#       torch.set_num_threads(N)   # N = 實體核心數，例如 8
#       device = torch.device('cpu')
#
#   Intel CPU 預設已透過 PyTorch 內建的 MKL-DNN（oneDNN）加速，
#   無需額外安裝。可用以下指令確認 MKL 已啟用：
#
#       python -c "import torch; print(torch.__config__.show())"
#       # 輸出中應可見 'USE_MKL=ON' 及 'USE_MKLDNN=ON'
#
#   若要進一步最佳化，可安裝 Intel Extension for PyTorch（IPEX）：
#
#       pip install intel-extension-for-pytorch
#
#       import intel_extension_for_pytorch as ipex
#       model = ipex.optimize(model)   # 套用 Intel 算子融合與量化優化
#       device = torch.device('cpu')   # IPEX 目前主力支援 CPU 路徑
#
# ============================================================

def get_device() -> torch.device:
    """
    Return the best available device in priority order:
      1. Intel/AMD GPU via DirectML (torch_directml)
      2. NVIDIA CUDA (if somehow available)
      3. CPU fallback

    若 DirectML 效益不理想，請參考上方「選項三」的替代方案說明。
    """
    try:
        import torch_directml
        if torch_directml.is_available():
            dml_device = torch_directml.device()
            print(f"[Device] DirectML — {torch_directml.device_name(0)}")
            return dml_device
    except ImportError:
        print("[Device] torch_directml not installed. "
              "Run: pip install torch-directml")

    if torch.cuda.is_available():
        print(f"[Device] CUDA — {torch.cuda.get_device_name(0)}")
        return torch.device('cuda')

    print("[Device] CPU (no GPU acceleration available)")
    return torch.device('cpu')


# ============================================================
# 1. Image Loading  (identical to denoise_torch.py)
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
# 2. UNet Architecture  (identical to denoise_torch.py)
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
    """4-level encoder-decoder UNet for Noise2Void (2D grayscale)."""

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
# 3. N2V Dataset with Blind-Spot Masking  (identical to denoise_torch.py)
# ============================================================

class N2VDataset(Dataset):
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
        P = self.patch_size
        corrupted = patch.copy()
        mask      = np.zeros((P, P), dtype=np.float32)
        flat_idx  = self.rng.choice(P * P, size=self.n_masked, replace=False)
        rows, cols = np.unravel_index(flat_idx, (P, P))
        rad = self.neighbor_radius
        dr_choices = np.arange(-rad, rad + 1)
        dc_choices = np.arange(-rad, rad + 1)
        for r, c in zip(rows, cols):
            while True:
                dr = int(self.rng.choice(dr_choices))
                dc = int(self.rng.choice(dc_choices))
                if dr != 0 or dc != 0:
                    break
            nr = int(np.clip(r + dr, 0, P - 1))
            nc = int(np.clip(c + dc, 0, P - 1))
            corrupted[r, c] = patch[nr, nc]
            mask[r, c]      = 1.0
        return corrupted, mask


# ============================================================
# 4. Training Loop  (DirectML-aware: pin_memory disabled)
# ============================================================

def train_n2v(
    model: nn.Module,
    image: np.ndarray,
    patch_size:     int   = 64,
    batch_size:     int   = 64,    # smaller default: iGPU has shared RAM
    num_epochs:     int   = 100,
    learning_rate:  float = 4e-4,
    val_percentage: float = 0.1,
    device: torch.device  = None,
) -> nn.Module:
    """
    Self-supervised N2V training on a single image.

    DirectML note: pin_memory is disabled (only works with CUDA).
    batch_size default is smaller (64) since iGPU shares system RAM.
    """
    if device is None:
        device = get_device()

    model = model.to(device)

    # Detect if running under DirectML to disable pin_memory
    is_cuda = hasattr(device, 'type') and device.type == 'cuda'

    patches_per_epoch = 2000
    n_val   = max(1, int(patches_per_epoch * val_percentage))
    n_train = patches_per_epoch - n_val

    train_ds = N2VDataset(image, patch_size=patch_size, num_patches=n_train, rng_seed=42)
    val_ds   = N2VDataset(image, patch_size=patch_size, num_patches=n_val,   rng_seed=99)

    # pin_memory=True only works with CUDA; must be False for DirectML/CPU
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=is_cuda)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)

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
# 5. Tiled Inference with Hann-Window Blending
#    DirectML note: hann_window computed on CPU, then converted to numpy.
#    DirectML does not support torch.hann_window on device directly.
# ============================================================

def predict_tiled(
    model: nn.Module,
    image: np.ndarray,
    tile_size:    Tuple[int, int] = (256, 256),
    tile_overlap: Tuple[int, int] = (48, 48),
    device: torch.device          = None,
) -> np.ndarray:
    """
    Run inference on large images by processing overlapping tiles.
    Hann window is always computed on CPU (DirectML compatibility).
    """
    if device is None:
        device = get_device()

    H, W      = image.shape
    th, tw    = tile_size
    oh, ow    = tile_overlap
    stride_h  = th - oh
    stride_w  = tw - ow

    assert th <= H and tw <= W, \
        f"tile_size {tile_size} must be <= image size {image.shape}"
    assert th % 8 == 0 and tw % 8 == 0, \
        f"tile_size dimensions must be divisible by 8, got {tile_size}"

    # Always compute Hann window on CPU (DirectML doesn't support hann_window on device)
    hann_h  = torch.hann_window(th, periodic=False).numpy()
    hann_w  = torch.hann_window(tw, periodic=False).numpy()
    hann_2d = np.outer(hann_h, hann_w).astype(np.float64)

    output_sum = np.zeros((H, W), dtype=np.float64)
    weight_sum = np.zeros((H, W), dtype=np.float64)

    row_starts = list(range(0, H - th + 1, stride_h))
    col_starts = list(range(0, W - tw + 1, stride_w))
    if row_starts[-1] + th < H:
        row_starts.append(H - th)
    if col_starts[-1] + tw < W:
        col_starts.append(W - tw)

    total_tiles = len(row_starts) * len(col_starts)
    processed   = 0

    model.eval()
    with torch.no_grad():
        for r0 in row_starts:
            for c0 in col_starts:
                tile_np = image[r0:r0 + th, c0:c0 + tw]
                tile_t  = torch.from_numpy(tile_np).unsqueeze(0).unsqueeze(0).to(device)

                # Move prediction back to CPU numpy (DirectML tensors need explicit .cpu())
                pred_np = model(tile_t).squeeze().cpu().numpy()

                output_sum[r0:r0 + th, c0:c0 + tw] += pred_np.astype(np.float64) * hann_2d
                weight_sum[r0:r0 + th, c0:c0 + tw] += hann_2d

                processed += 1
                if processed % 10 == 0 or processed == total_tiles:
                    print(f"  Inference: {processed}/{total_tiles} tiles")

    denoised = (output_sum / np.maximum(weight_sum, 1e-8)).astype(np.float32)
    return denoised


# ============================================================
# 6. Save Outputs  (identical to denoise_torch.py)
# ============================================================

def save_outputs(
    image:    np.ndarray,
    denoised: np.ndarray,
    img_min:  float,
    img_max:  float,
    tif_path: str = "denoised_sem_directml.tif",
    png_path: str = "denoising_result_directml.png",
) -> None:
    denoised_original = (denoised * (img_max - img_min) + img_min).astype(np.float32)
    tifffile.imwrite(tif_path, denoised_original)
    print(f"Saved: {tif_path},  range: [{denoised_original.min():.3f}, {denoised_original.max():.3f}]")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image,    cmap='gray')
    axes[0].set_title('Original SEM Image')
    axes[0].axis('off')
    axes[1].imshow(denoised, cmap='gray')
    axes[1].set_title('N2V Denoised (DirectML)')
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
    image_path:   str             = "test_sem.tif",
    patch_size:   int             = 64,
    batch_size:   int             = 64,
    num_epochs:   int             = 100,
    tile_size:    Tuple[int, int] = (256, 256),
    tile_overlap: Tuple[int, int] = (48, 48),
) -> None:
    """Full N2V pipeline: load -> train -> predict -> save."""
    device = get_device()

    image, img_min, img_max = load_sem_image(image_path)
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
        device=device,
    )

    save_outputs(image, denoised, img_min, img_max)


if __name__ == '__main__':
    main()

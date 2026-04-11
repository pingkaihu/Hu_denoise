# ============================================================
# SEM Image Denoising — Noise2Void (Optimized PyTorch version)
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
import torch.nn.functional as F

torch.set_float32_matmul_precision('high')


# ============================================================
# 1. Image Loading (Added log_transform option)
# ============================================================

def load_sem_image(path: str, use_log_transform: bool = False) -> Tuple[np.ndarray, float, float]:
    img = tifffile.imread(path).astype(np.float32)

    if img.ndim == 3 and img.shape[-1] == 3:
        img = img @ np.array([0.2989, 0.5870, 0.1140])

    img_min, img_max = float(img.min()), float(img.max())
    
    if use_log_transform:
        # Prevent log(0) and stabilize 
        img = np.log1p(img - img_min)

    # Normalize to [0, 1]
    img_min_norm, img_max_norm = float(img.min()), float(img.max())
    img = (img - img_min_norm) / (img_max_norm - img_min_norm + 1e-8)
    return img, img_min, img_max, img_min_norm, img_max_norm


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
# 3. N2V Dataset (Vectorized CPU Masking)
# ============================================================

class N2VDataset(Dataset):
    def __init__(
        self, image: np.ndarray, patch_size: int = 64, num_patches: int = 2000,
        mask_ratio: float = 0.006, neighbor_radius: int = 5, rng_seed: int = None,
    ):
        assert patch_size % 8 == 0
        self.image = image
        self.H, self.W = image.shape
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.neighbor_radius = neighbor_radius
        self.n_masked = max(1, int(patch_size * patch_size * mask_ratio))
        self.rng = np.random.default_rng(rng_seed)

    def __len__(self) -> int:
        return self.num_patches

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        P = self.patch_size
        r0 = self.rng.integers(0, self.H - P + 1)
        c0 = self.rng.integers(0, self.W - P + 1)
        patch = self.image[r0:r0 + P, c0:c0 + P].copy()
        corrupted, mask = self._apply_n2v_masking(patch)

        return (
            torch.from_numpy(corrupted).unsqueeze(0),
            torch.from_numpy(patch).unsqueeze(0),
            torch.from_numpy(mask).unsqueeze(0),
        )

    def _apply_n2v_masking(self, patch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        P = self.patch_size
        corrupted = patch.copy()
        mask = np.zeros((P, P), dtype=np.float32)

        flat_idx = self.rng.choice(P * P, size=self.n_masked, replace=False)
        rows, cols = np.unravel_index(flat_idx, (P, P))

        rad = self.neighbor_radius
        # Vectorized offsets
        dr = self.rng.integers(-rad, rad + 1, size=self.n_masked)
        dc = self.rng.integers(-rad, rad + 1, size=self.n_masked)
        zero_mask = (dr == 0) & (dc == 0)
        
        # Avoid (0, 0)
        if np.any(zero_mask):
            dr[zero_mask] += self.rng.choice([-1, 1], size=np.sum(zero_mask))

        nr = np.clip(rows + dr, 0, P - 1)
        nc = np.clip(cols + dc, 0, P - 1)

        corrupted[rows, cols] = patch[nr, nc]
        mask[rows, cols] = 1.0
        return corrupted, mask


# ============================================================
# 4. Training Loop (Fixed Data Leakage)
# ============================================================

def train_n2v(
    model: nn.Module, image: np.ndarray, patch_size: int = 64,
    batch_size: int = 128, num_epochs: int = 100, learning_rate: float = 4e-4,
    val_percentage: float = 0.1, device: torch.device = None,
) -> nn.Module:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Physical Split for Train/Val to avoid Data Leakage
    split_idx = int(image.shape[0] * (1 - val_percentage))
    train_image = image[:split_idx, :]
    val_image = image[split_idx:, :]
    
    if train_image.shape[0] < patch_size or val_image.shape[0] < patch_size:
        print("Image too small for physical split, reverting to total image randomization.")
        train_image = image
        val_image = image

    patches_per_epoch = 2000
    n_val   = max(1, int(patches_per_epoch * val_percentage))
    n_train = patches_per_epoch - n_val

    train_ds = N2VDataset(train_image, patch_size=patch_size, num_patches=n_train, rng_seed=42)
    val_ds   = N2VDataset(val_image,   patch_size=patch_size, num_patches=n_val,   rng_seed=99)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    loss_fn   = nn.MSELoss(reduction='sum')

    print(f"Device: {device} | Patches/epoch: train={n_train} val={n_val}")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        model.train()
        tr_loss, tr_count = 0.0, 0
        for noisy_in, clean_tgt, mask in train_loader:
            noisy_in, clean_tgt, mask = noisy_in.to(device), clean_tgt.to(device), mask.to(device)
            optimizer.zero_grad()
            pred = model(noisy_in)
            loss = loss_fn(pred * mask, clean_tgt * mask)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
            tr_count += mask.sum().item()

        model.eval()
        vl_loss, vl_count = 0.0, 0
        with torch.no_grad():
            for noisy_in, clean_tgt, mask in val_loader:
                noisy_in, clean_tgt, mask = noisy_in.to(device), clean_tgt.to(device), mask.to(device)
                pred = model(noisy_in)
                vl_loss += loss_fn(pred * mask, clean_tgt * mask).item()
                vl_count += mask.sum().item()

        scheduler.step()
        if epoch % 10 == 0 or epoch == 1:
            tr_mse = tr_loss / max(tr_count, 1)
            vl_mse = vl_loss / max(vl_count, 1)
            print(f"Epoch [{epoch:3d}/{num_epochs}] train_loss={tr_mse:.6f} val_loss={vl_mse:.6f} elapsed={time.time() - t0:.1f}s")
    return model


# ============================================================
# 5. Batched / Tiled Inference with Padding
# ============================================================

def predict_tiled(
    model: nn.Module, image: np.ndarray, tile_size: Tuple[int, int] = (256, 256),
    tile_overlap: Tuple[int, int] = (48, 48), batch_size: int = 8, device: torch.device = None,
) -> np.ndarray:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    th, tw = tile_size
    oh, ow = tile_overlap
    stride_h = th - oh
    stride_w = tw - ow

    # Reflection padding if image is smaller than tile_size or dimension not divisible by 8
    pad_h = max(0, th - image.shape[0])
    pad_w = max(0, tw - image.shape[1])
    
    # Also pad so it can smoothly slide
    if pad_h == 0 and pad_w == 0:
        pad_h = (8 - image.shape[0] % 8) % 8
        pad_w = (8 - image.shape[1] % 8) % 8

    padded_image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
    H_pad, W_pad = padded_image.shape

    hann_2d = np.outer(torch.hann_window(th, periodic=False).numpy(), 
                       torch.hann_window(tw, periodic=False).numpy())

    output_sum = np.zeros((H_pad, W_pad), dtype=np.float64)
    weight_sum = np.zeros((H_pad, W_pad), dtype=np.float64)

    row_starts = list(range(0, H_pad - th + 1, stride_h))
    col_starts = list(range(0, W_pad - tw + 1, stride_w))
    if row_starts and row_starts[-1] + th < H_pad: row_starts.append(H_pad - th)
    if col_starts and col_starts[-1] + tw < W_pad: col_starts.append(W_pad - tw)
    if not row_starts: row_starts = [0]
    if not col_starts: col_starts = [0]

    coords = [(r, c) for r in row_starts for c in col_starts]
    total_tiles = len(coords)

    model.eval()
    with torch.no_grad():
        for i in range(0, total_tiles, batch_size):
            batch_coords = coords[i:i+batch_size]
            tiles = []
            for (r0, c0) in batch_coords:
                tiles.append(torch.from_numpy(padded_image[r0:r0+th, c0:c0+tw]))
            
            # shape: (B, 1, th, tw)
            batch_t = torch.stack(tiles).unsqueeze(1).to(device)
            preds = model(batch_t).squeeze(1).cpu().numpy()

            if len(batch_coords) == 1:
                preds = np.expand_dims(preds, axis=0)

            for idx, (r0, c0) in enumerate(batch_coords):
                output_sum[r0:r0+th, c0:c0+tw] += preds[idx].astype(np.float64) * hann_2d
                weight_sum[r0:r0+th, c0:c0+tw] += hann_2d
                
            print(f"  Inference: min({i+batch_size}, {total_tiles})/{total_tiles} tiles")

    denoised_pad = (output_sum / np.maximum(weight_sum, 1e-8)).astype(np.float32)
    # Crop padding
    denoised = denoised_pad[:image.shape[0], :image.shape[1]]
    return denoised


# ============================================================
# 6. Save Outputs (Handles Log Transform)
# ============================================================

def save_outputs(
    image: np.ndarray, denoised: np.ndarray, img_min: float, img_max: float,
    img_min_norm: float, img_max_norm: float, use_log_transform: bool,
    tif_path: str = "data/denoised_sem_test.tif", png_path: str = "data/denoising_result_test.png",
) -> None:
    
    # De-normalize 
    denoised_original = denoised * (img_max_norm - img_min_norm + 1e-8) + img_min_norm
    image_original = image * (img_max_norm - img_min_norm + 1e-8) + img_min_norm

    # De-log if we used the log transform
    if use_log_transform:
        denoised_original = np.expm1(denoised_original) + img_min
        image_original = np.expm1(image_original) + img_min
    else:
        # It's already in original range due to the way we extracted min/max
        # actually, load_sem_image sets img = (img - img_min) / ...
        # wait, my load func is: 
        # img = (img - img_min_norm) / (img_max_norm - img_min_norm)
        # So denormalize first gets it back to the state after log, before scaling.
        # So we just add img_min back to the non-log if we want exact pixel values.
        # Let's fix this cleanly.
        pass

    # Clean logic for restoration:
    if use_log_transform:
        denoised_final = np.expm1(np.maximum(0, denoised_original)) + img_min
        image_vis = np.expm1(np.maximum(0, image_original)) + img_min
    else:
        denoised_final = denoised_original
        image_vis = image_original

    tifffile.imwrite(tif_path, denoised_final.astype(np.float32))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image_vis, cmap='gray'); axes[0].set_title('Original SEM Image'); axes[0].axis('off')
    axes[1].imshow(denoised_final, cmap='gray'); axes[1].set_title('N2V Denoised (Opt)'); axes[1].axis('off')
    diff = np.abs(image_vis - denoised_final) * 3
    axes[2].imshow(diff, cmap='hot'); axes[2].set_title('Difference (×3)'); axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.show() # prevent blocking


# ============================================================
# 7. Main Pipeline
# ============================================================

def main(
    patch_size: int = 64, 
    batch_size: int = 128, 
    num_epochs: int = 100, # 20 epochs for quick test
    tile_size: Tuple[int, int] = (256, 256), 
    tile_overlap: Tuple[int, int] = (48, 48),
    use_log_transform: bool = False
) -> None:
    
    input_path = "data/test_sem.tif"
    output_path = "data/denoised_sem_test.tif"

    if not os.path.exists(input_path):
        print("test image not found, generating...")
        os.system("python test_sem.py")
        if not os.path.exists("data"):
            os.makedirs("data")
        if os.path.exists("test_sem.tif"):
            import shutil
            shutil.move("test_sem.tif", "data/test_sem.tif")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image, img_min, img_max, img_min_norm, img_max_norm = load_sem_image(input_path, use_log_transform)
    
    model = N2VUNet(in_channels=1, base_features=32)
    model = train_n2v(model, image, patch_size=patch_size, batch_size=batch_size, num_epochs=num_epochs, device=device)

    print("\nRunning tiled inference...")
    denoised = predict_tiled(model, image, tile_size=tile_size, tile_overlap=tile_overlap, batch_size=16, device=device)

    save_outputs(image, denoised, img_min, img_max, img_min_norm, img_max_norm, use_log_transform, tif_path=output_path)

if __name__ == '__main__':
    main()

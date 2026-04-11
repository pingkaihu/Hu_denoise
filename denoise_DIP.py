# ============================================================
# SEM Image Denoising — Deep Image Prior (DIP, pure PyTorch)
# ============================================================
# Paper: "Deep Image Prior" — Ulyanov, Vedaldi, Lempitsky, CVPR 2018
#        https://dmitryulyanov.github.io/deep_image_prior
#
# Core idea (核心概念):
#   A randomly-initialized CNN is trained ONLY on the noisy input image —
#   no dataset, no clean reference images needed. The CNN structure itself
#   acts as an implicit prior over natural images (spectral bias: the network
#   learns low-frequency structure first and high-frequency noise last).
#   Early stopping halts training before the network overfits to noise,
#   recovering a clean estimate of the underlying image.
#
#   隨機初始化的 CNN 僅對含噪影像進行訓練，無需資料集或乾淨參考影像。
#   CNN 結構本身作為自然影像的隱式先驗（頻譜偏置：網路先學習低頻結構，
#   最後才學習高頻噪聲）。早停在網路過擬合噪聲前停止訓練，還原乾淨影像。
#
# Architecture differences from N2V (與 N2V 的架構差異):
#   - Encoder uses strided Conv2d (stride=2) instead of MaxPool — smoother
#     gradient flow and better spectral bias properties
#     編碼器使用步進卷積（stride=2）而非 MaxPool，梯度流更平滑
#   - InstanceNorm instead of BatchNorm — correct for batch_size=1
#     使用 InstanceNorm 而非 BatchNorm，適合 batch_size=1 的情況
#   - Narrow 4-channel skip connections — intentionally constrained so the
#     decoder cannot bypass the bottleneck's implicit regularization
#     窄化的 4 通道跳接連線，防止解碼器繞過瓶頸層的隱式正則化
#   - Sigmoid output head — constrains output to [0,1], consistent with
#     normalized input target
#     Sigmoid 輸出頭，將輸出限制在 [0,1]，與正規化輸入目標一致
#   - No Dataset/DataLoader needed — a single forward pass on the full image
#     無需 Dataset/DataLoader，直接對整張影像做前向傳播
#
# Usage (使用方式):
#   python test_sem.py      # generate data/test_sem.tif if not present
#   python denoise_DIP.py   # train + denoise -> data/denoised_sem_DIP.tif
# ============================================================
# Requirements: torch>=2.0.0  tifffile  matplotlib  numpy
# ============================================================

import argparse
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import copy
import time
from typing import Tuple, Optional

import numpy as np
import tifffile
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.set_float32_matmul_precision('high')  # 啟用 Tensor Core，提升 RTX 訓練速度


# ============================================================
# 0. Parameters  (adjust these at the top for easy tuning)
# ============================================================

INPUT_PATH  = "data/test_sem.tif"
OUTPUT_PATH = "data/denoised_sem_DIP.tif"
PNG_PATH    = "data/denoising_DIP_result.png"

# --- Architecture ---
Z_CHANNELS    = 32    # fixed random noise input channels  (固定噪聲輸入通道數)
NUM_CHANNELS  = 128   # encoder/decoder channel width       (編解碼器通道寬度)
NUM_LEVELS    = 5     # encoder depth: number of stride-2 downsampling steps (編碼器深度)
SKIP_CHANNELS = 4     # narrow skip connection width        (窄化跳接通道數)

# --- Training ---
NUM_ITERATIONS = 3000   # max gradient-descent iterations  (最大梯度下降迭代數)
LEARNING_RATE  = 0.01   # Adam learning rate
REG_NOISE_STD  = 0.03   # std of per-iteration z perturbation (z 擾動標準差)
                         # Adding small noise to z each iter prevents exact
                         # z memorization and acts as a regularizer.
                         # 每次迭代對 z 加入小噪聲，防止精確記憶並作為正則化手段。

# --- Early stopping  (早停參數) ---
MIN_ITERATIONS = 500    # never stop before this many iterations  (最少迭代次數)
EMA_ALPHA      = 0.99   # exponential moving average decay factor  (EMA 衰減係數)
                         # Higher = smoother, slower to react. 0.99 ≈ 100-iter average.
                         # 越高越平滑但反應越慢；0.99 ≈ 約 100 次迭代的滑動平均
PATIENCE       = 50     # consecutive iterations with loss > EMA before stopping
                         # (連續超過 EMA 的次數上限，超過則觸發早停)

# --- Large image auto-scaling ---
MAX_SIDE = 2048   # if max(H, W) > MAX_SIDE, auto-reduce NUM_CHANNELS to 64
                  # DIP requires a single full-image forward pass — no tiling.
                  # Channel reduction keeps VRAM under ~6 GB for images up to ~3000 px.
                  # (若影像過大自動縮減通道數，保持顯存在合理範圍)


# ============================================================
# 1. Image Loading  (identical to denoise_N2V.py)
# ============================================================

def load_sem_image(path: str) -> Tuple[np.ndarray, float, float]:
    """Load SEM image, normalize to float32 [0, 1] grayscale numpy array.
    Also returns original min/max for restoring pixel values after denoising.
    載入 SEM 影像，正規化為 float32 [0, 1] 灰階陣列，同時返回原始範圍供還原使用。"""
    img = tifffile.imread(path).astype(np.float32)

    if img.ndim == 3 and img.shape[-1] == 3:
        img = img @ np.array([0.2989, 0.5870, 0.1140])

    img_min, img_max = float(img.min()), float(img.max())
    img = (img - img_min) / (img_max - img_min + 1e-8)
    return img, img_min, img_max


# ============================================================
# 2. DIP Architecture
# ============================================================

class DIPConvBlock(nn.Module):
    """Single Conv2d + InstanceNorm2d(affine=True) + LeakyReLU(0.1) block.
    單層卷積 + 實例正規化 + LeakyReLU 區塊。

    Used for both encoder (stride=2, downsampling) and decoder (stride=1).
    InstanceNorm is correct here because DIP always runs with batch_size=1;
    BatchNorm would compute trivial statistics and provide no normalization.
    DIP 始終以 batch_size=1 運行，BatchNorm 統計量退化；InstanceNorm 是正確選擇。
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride,
                      padding=1, bias=True),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _DIPSkipBlock(nn.Module):
    """Narrow 1×1 skip connection: C → skip_channels.
    窄化的 1×1 跳接連線，刻意限制通道數以防止解碼器繞過瓶頸層的隱式正則化。"""

    def __init__(self, in_ch: int, skip_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, skip_ch, kernel_size=1, bias=True),
            nn.InstanceNorm2d(skip_ch, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DIPNet(nn.Module):
    """
    Deep Image Prior encoder-decoder generator network.
    深度影像先驗編解碼器生成網路。

    Architecture overview (架構概覽):
      Input z : (1, Z_CHANNELS, H_pad, W_pad)  — fixed uniform noise [0, 0.1]

      Encoder : num_levels × DIPConvBlock(stride=2)
                Channels: Z_CHANNELS → NUM_CHANNELS at enc[0], then NUM_CHANNELS
                throughout. Uses strided conv (not MaxPool) for smoother gradient
                flow and better spectral bias properties.
                使用步進卷積（非 MaxPool）以獲得更平滑的梯度流和更好的頻譜偏置特性。

      Skips   : (num_levels - 1) narrow 1×1 skip connections from enc[0..n-2].
                The bottleneck (enc[-1]) intentionally has NO skip — this forces
                the decoder to reconstruct through the low-dimensional bottleneck,
                strengthening the implicit prior.
                瓶頸層（最深的編碼器層）刻意不設跳接，強迫解碼器通過低維瓶頸重建，
                增強隱式先驗效果。

      Decoder : num_levels × bilinear Upsample(×2) + DIPConvBlock(stride=1)
                dec[0..n-2] concatenate the reversed skip features.
                dec[-1] has no skip (returns to input spatial resolution).

      Head    : Conv2d(NUM_CHANNELS → 1, k=1) + Sigmoid
                Sigmoid constrains output to [0, 1], consistent with normalized target.
                Sigmoid 將輸出限制在 [0,1]，與正規化後的目標影像一致。

      Output  : (1, 1, H_pad, W_pad) ∈ [0, 1]
                Caller crops to original (H, W) after inference.
    """

    def __init__(
        self,
        z_channels:    int = Z_CHANNELS,
        num_channels:  int = NUM_CHANNELS,
        num_levels:    int = NUM_LEVELS,
        skip_channels: int = SKIP_CHANNELS,
    ):
        super().__init__()
        self.num_levels    = num_levels
        self.skip_channels = skip_channels
        C = num_channels
        S = skip_channels

        # ---- Encoder: num_levels strided-conv blocks ----
        # enc[0] accepts z_channels; enc[1..] accept C channels
        enc_blocks = []
        in_ch = z_channels
        for _ in range(num_levels):
            enc_blocks.append(DIPConvBlock(in_ch, C, stride=2))
            in_ch = C
        self.encoder = nn.ModuleList(enc_blocks)

        # ---- Skip connections: (num_levels - 1) narrow 1×1 projections ----
        # skip[i] is applied to encoder_feats[i], for i in 0..num_levels-2
        # (bottleneck encoder_feats[-1] gets no skip)
        self.skips = nn.ModuleList(
            [_DIPSkipBlock(C, S) for _ in range(num_levels - 1)]
        )

        # ---- Decoder: num_levels blocks ----
        # dec[0..num_levels-2] receive concat(upsampled, skip) → C+S channels in
        # dec[num_levels-1]    receives only upsampled → C channels in (no skip)
        dec_blocks = []
        for i in range(num_levels):
            in_ch = C + S if i < num_levels - 1 else C
            dec_blocks.append(DIPConvBlock(in_ch, C, stride=1))
        self.decoder = nn.ModuleList(dec_blocks)

        # ---- Output head ----
        self.head = nn.Sequential(
            nn.Conv2d(C, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # --- Encode ---
        encoder_feats = []
        x = z
        for enc_block in self.encoder:
            x = enc_block(x)
            encoder_feats.append(x)
        # encoder_feats[-1] is the bottleneck (no skip from here)

        # --- Decode (reverse order, skip connections matched in reverse) ---
        x = encoder_feats[-1]
        for i in range(self.num_levels):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

            # Skip index: dec step 0 uses enc[n-2], step 1 uses enc[n-3], ..., step n-2 uses enc[0]
            # Step n-1 (last) has no skip.
            skip_idx = self.num_levels - 2 - i
            if skip_idx >= 0:
                skip_feat = self.skips[skip_idx](encoder_feats[skip_idx])
                x = torch.cat([x, skip_feat], dim=1)

            x = self.decoder[i](x)

        return self.head(x)


# ============================================================
# 3. Padding Utility
# ============================================================

def pad_to_multiple(
    image: np.ndarray,
    multiple: int,
) -> Tuple[np.ndarray, int, int]:
    """Reflection-pad H and W to the nearest multiple of `multiple`.
    以反射填充將 H 和 W 補齊為 `multiple` 的最小整數倍。

    For NUM_LEVELS=5, multiple=32 (2^5). This ensures all 5 stride-2
    downsampling steps divide evenly, avoiding shape mismatches in the decoder.
    對於 NUM_LEVELS=5，multiple=32；確保 5 次步進=2 的下採樣步驟均能整除。

    Returns (padded_image, pad_h, pad_w).
    """
    H, W = image.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
    return image, pad_h, pad_w


# ============================================================
# 4. DIP Training Loop
# ============================================================

def train_dip(
    model:          nn.Module,
    image:          np.ndarray,           # (H, W) float32 normalized [0, 1]
    num_iterations: int   = NUM_ITERATIONS,
    learning_rate:  float = LEARNING_RATE,
    reg_noise_std:  float = REG_NOISE_STD,
    z_channels:     int   = Z_CHANNELS,
    min_iterations: int   = MIN_ITERATIONS,
    ema_alpha:      float = EMA_ALPHA,
    patience:       int   = PATIENCE,
    num_levels:     int   = NUM_LEVELS,
    log_every:      int   = 100,
    device:         Optional[torch.device] = None,
) -> Tuple[nn.Module, np.ndarray]:
    """
    Train DIP by gradient descent on a single noisy image.
    對單張含噪影像進行梯度下降訓練。

    Unlike N2V, there is no dataset or dataloader — the generator network
    maps a fixed random noise tensor z directly to the denoised output.
    與 N2V 不同，此處無需資料集或 DataLoader；生成器網路將固定的隨機噪聲張量
    z 直接映射到去噪輸出。

    Early stopping strategy (早停策略):
      - EMA tracks a smoothed version of the loss curve
        EMA 追蹤損失曲線的平滑版本
      - When raw_loss > ema_loss for `patience` consecutive iterations,
        the network has likely started overfitting to noise → stop
        當 raw_loss 連續 patience 次超過 ema_loss，表示網路開始過擬合噪聲 → 停止
      - Additionally, the best-snapshot at the lowest raw loss is always
        restored at the end, even if early stopping triggers a few iters late
        此外，始終還原最低原始損失時的最佳快照，即使早停稍微延遲觸發也無影響

    Returns (model_with_best_weights, denoised_float32_numpy_H_W).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    H, W = image.shape
    multiple = 2 ** num_levels
    padded, pad_h, pad_w = pad_to_multiple(image, multiple)
    H_pad, W_pad = padded.shape

    # Fixed random input z — uniform [0, 0.1] as recommended in the original paper.
    # z is created on device and never modified; only z_perturbed changes each iter.
    # 固定隨機輸入 z，使用均勻分佈 [0, 0.1]（原論文推薦）。
    # z_fixed 在整個訓練中保持不變；每次迭代僅 z_perturbed 不同。
    z_fixed = torch.rand(1, z_channels, H_pad, W_pad, device=device) * 0.1

    # Target: noisy image on device  (1, 1, H_pad, W_pad)
    target = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn   = nn.MSELoss()

    ema_loss        = None          # initialized from first loss value
    best_loss       = float('inf')
    best_weights    = copy.deepcopy(model.state_dict())
    patience_counter = 0

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}  |  Model parameters: {n_params:,}")
    print(f"Image: {H}×{W} → padded {H_pad}×{W_pad}  "
          f"(pad_h={pad_h}, pad_w={pad_w})")
    print(f"Training: max_iterations={num_iterations}  lr={learning_rate}  "
          f"reg_noise_std={reg_noise_std}")
    print(f"Early stopping: min_iters={min_iterations}  "
          f"ema_alpha={ema_alpha}  patience={patience}")

    t0 = time.time()

    for iteration in range(1, num_iterations + 1):
        model.train()

        # Add small Gaussian perturbation to z each iteration.
        # This prevents the network from memorizing a specific z and acts as
        # a lightweight noise regularizer, improving early-stopping stability.
        # 每次迭代對 z 加入小高斯擾動，防止網路記憶特定 z，提升早停穩定性。
        z_perturbed = z_fixed + torch.randn_like(z_fixed) * reg_noise_std

        optimizer.zero_grad()
        pred = model(z_perturbed)
        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()

        raw_loss = loss.item()

        # Best-snapshot tracking: always save the state at the lowest loss seen.
        # The final output is produced from this snapshot (not the stopped-at state).
        # 最佳快照追蹤：始終儲存損失最低時的狀態，最終輸出由此快照生成。
        if raw_loss < best_loss:
            best_loss    = raw_loss
            best_weights = copy.deepcopy(model.state_dict())

        # Exponential Moving Average of loss for early stopping.
        # Init from first value (not 0) to avoid large initial gap.
        # EMA 初始化為第一次損失值（非 0），避免初始差距過大影響早停判斷。
        if ema_loss is None:
            ema_loss = raw_loss
        else:
            ema_loss = ema_alpha * ema_loss + (1.0 - ema_alpha) * raw_loss

        # Early stopping check (only after lockout period)
        if iteration >= min_iterations:
            if raw_loss > ema_loss:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at iteration {iteration}  "
                          f"(loss={raw_loss:.6f} > ema={ema_loss:.6f})")
                    break
            else:
                patience_counter = 0

        if iteration % log_every == 0 or iteration == 1:
            elapsed = time.time() - t0
            print(f"  iter {iteration:4d}/{num_iterations}  "
                  f"loss={raw_loss:.6f}  ema={ema_loss:.6f}  "
                  f"elapsed={elapsed:.1f}s")

    # Restore best weights (lowest-loss snapshot)
    # 還原最佳權重（損失最低的快照）
    model.load_state_dict(best_weights)
    print(f"\nRestored best snapshot  (best_loss={best_loss:.6f})")

    # Final inference with clean z_fixed (no perturbation for deterministic output)
    # 最終推論使用乾淨的 z_fixed（無擾動，確保輸出確定性）
    model.eval()
    with torch.no_grad():
        denoised_t = model(z_fixed)  # (1, 1, H_pad, W_pad)

    denoised = denoised_t.squeeze().cpu().numpy()   # (H_pad, W_pad)
    denoised = np.clip(denoised[:H, :W], 0.0, 1.0)  # crop + clamp

    return model, denoised


# ============================================================
# 5. Save Outputs  (same pattern as denoise_N2V.py)
# ============================================================

def save_outputs(
    image:    np.ndarray,
    denoised: np.ndarray,
    img_min:  float,
    img_max:  float,
    tif_path: str = OUTPUT_PATH,
    png_path: str = PNG_PATH,
) -> None:
    """Save denoised TIF (original value range) and side-by-side comparison PNG.
    儲存去噪後的 TIF（原始數值範圍）及並排對比 PNG（原始 | DIP 去噪 | 差異×3）。"""
    os.makedirs(os.path.dirname(tif_path) or '.', exist_ok=True)

    denoised_original = (denoised * (img_max - img_min) + img_min).astype(np.float32)
    tifffile.imwrite(tif_path, denoised_original)
    print(f"Saved: {tif_path},  "
          f"range: [{denoised_original.min():.3f}, {denoised_original.max():.3f}]")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image,    cmap='gray')
    axes[0].set_title('Original SEM Image')
    axes[0].axis('off')

    axes[1].imshow(denoised, cmap='gray')
    axes[1].set_title('DIP Denoised')
    axes[1].axis('off')

    diff = np.abs(image - denoised) * 3
    axes[2].imshow(diff, cmap='hot')
    axes[2].set_title('Difference (×3)')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {png_path}")


# ============================================================
# 6. Main Pipeline
# ============================================================

def main() -> None:
    """Full DIP denoising pipeline: load → train → save."""
    parser = argparse.ArgumentParser(
        description="DIP SEM denoiser: Deep Image Prior, no dataset required."
    )
    parser.add_argument('--input',          type=str,   default=INPUT_PATH,
                        help='Path to input .tif/.tiff/.png image')
    parser.add_argument('--output',         type=str,   default='',
                        help=f'Path to output .tif (default: {OUTPUT_PATH})')
    parser.add_argument('--num_channels',   type=int,   default=NUM_CHANNELS,
                        help='Encoder/decoder width; reduce to 64 if VRAM is tight or image >2048px')
    parser.add_argument('--num_levels',     type=int,   default=NUM_LEVELS,
                        help='Encoder depth (stride-2 steps); auto-reduced for small images')
    parser.add_argument('--num_iterations', type=int,   default=NUM_ITERATIONS,
                        help='Max iterations (early stopping may halt sooner); 1000 for fast preview')
    parser.add_argument('--lr',             type=float, default=LEARNING_RATE,
                        help='Adam learning rate')
    parser.add_argument('--reg_noise_std',  type=float, default=REG_NOISE_STD,
                        help='z-perturbation std: 0.03 standard, 0.05 heavy noise, 0.01 smooth')
    parser.add_argument('--min_iterations', type=int,   default=MIN_ITERATIONS,
                        help='Never stop before this many iterations')
    parser.add_argument('--patience',       type=int,   default=PATIENCE,
                        help='Consecutive iters with loss > EMA before early stop')
    args = parser.parse_args()

    input_path     = args.input
    output_path    = args.output or OUTPUT_PATH
    png_path       = PNG_PATH
    num_channels   = args.num_channels
    num_levels     = args.num_levels
    num_iterations = args.num_iterations
    learning_rate  = args.lr
    reg_noise_std  = args.reg_noise_std
    min_iterations = args.min_iterations
    patience       = args.patience

    t_start = time.time()
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Load ---
    print(f"Loading: {input_path}")
    image, img_min, img_max = load_sem_image(input_path)
    H, W = image.shape
    print(f"Image shape: {H}×{W},  "
          f"pixel range: [{img_min:.3f}, {img_max:.3f}]")

    # --- Auto-scale architecture for large images ---
    # DIP processes the full image in a single forward pass (no tiling).
    # For very large images, reduce channel count to keep VRAM manageable.
    # DIP 對整張影像做單次前向傳播（無分塊）；對大影像縮減通道數以控制顯存。
    if max(H, W) > MAX_SIDE:
        print(f"Warning: image side {max(H, W)} > MAX_SIDE={MAX_SIDE}. "
              f"Auto-reducing num_channels: {num_channels} → 64")
        num_channels = 64

    # --- Auto-scale architecture for tiny images ---
    # With NUM_LEVELS=5, the bottleneck is H/32 × W/32. For images <64px,
    # reduce num_levels so the bottleneck stays ≥ 2px.
    # 瓶頸空間大小為 H/32；對小於 64px 的影像縮減 num_levels 保持瓶頸 ≥ 2px。
    max_levels = int(np.floor(np.log2(min(H, W))))
    if num_levels > max_levels:
        print(f"Warning: image too small for {num_levels} levels. "
              f"Auto-reducing: {num_levels} → {max_levels}")
        num_levels = max_levels

    # --- Build model ---
    model = DIPNet(
        z_channels=Z_CHANNELS,
        num_channels=num_channels,
        num_levels=num_levels,
        skip_channels=SKIP_CHANNELS,
    )

    # --- Train ---
    print("\n--- DIP Training ---")
    _, denoised = train_dip(
        model, image,
        num_iterations=num_iterations,
        learning_rate=learning_rate,
        reg_noise_std=reg_noise_std,
        z_channels=Z_CHANNELS,
        min_iterations=min_iterations,
        ema_alpha=EMA_ALPHA,
        patience=patience,
        num_levels=num_levels,
        device=device,
    )

    # --- Save ---
    print("\n--- Saving outputs ---")
    save_outputs(image, denoised, img_min, img_max,
                 tif_path=output_path, png_path=png_path)

    print(f"\nTotal elapsed: {time.time() - t_start:.1f}s")


if __name__ == '__main__':
    main()

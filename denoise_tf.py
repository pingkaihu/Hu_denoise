# ============================================================
# SEM Image Denoising — Noise2Void (TensorFlow/Keras, no careamics)
# ============================================================
# Requirements: tensorflow>=2.10  tifffile  matplotlib  numpy
# Usage:
#   python test_sem.py       # generate synthetic test image
#   python denoise_tf.py     # train + denoise -> denoised_sem.tif
# ============================================================

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import time
from typing import Tuple

import numpy as np
import tifffile
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# 1. Image Loading  (identical to denoise.py / denoise_torch.py)
# ============================================================

def load_sem_image(path: str) -> Tuple[np.ndarray, float, float]:
    """Load SEM image, normalize to float32 [0, 1] grayscale numpy array.
    Also returns original min/max for restoring pixel values after denoising."""
    img = tifffile.imread(path).astype(np.float32)

    # Convert RGB to grayscale if needed
    if img.ndim == 3 and img.shape[-1] == 3:
        img = img @ np.array([0.2989, 0.5870, 0.1140])

    # Preserve original range, normalize to [0, 1]
    img_min, img_max = float(img.min()), float(img.max())
    img = (img - img_min) / (img_max - img_min + 1e-8)
    return img, img_min, img_max


# ============================================================
# 2. N2V Blind-Spot Masking  (standalone, used by the generator)
# ============================================================

def apply_n2v_masking(
    patch: np.ndarray,
    n_masked: int,
    neighbor_radius: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    N2V masking: replace n_masked random pixels with neighbor values.

    Returns (corrupted_patch, mask) — mask is 1.0 at masked positions.
    Masked pixels are replaced with a random neighbor value (NOT zeros)
    to prevent the network from detecting masked positions by intensity.
    """
    P = patch.shape[0]
    corrupted = patch.copy()
    mask      = np.zeros((P, P), dtype=np.float32)

    flat_idx  = rng.choice(P * P, size=n_masked, replace=False)
    rows, cols = np.unravel_index(flat_idx, (P, P))

    rad = neighbor_radius
    for r, c in zip(rows, cols):
        while True:
            dr = int(rng.integers(-rad, rad + 1))
            dc = int(rng.integers(-rad, rad + 1))
            if dr != 0 or dc != 0:
                break
        nr = int(np.clip(r + dr, 0, P - 1))
        nc = int(np.clip(c + dc, 0, P - 1))
        corrupted[r, c] = patch[nr, nc]
        mask[r, c]      = 1.0

    return corrupted, mask


# ============================================================
# 3. UNet Architecture — Keras Functional API (channels-last)
# ============================================================

def double_conv_block(x: tf.Tensor, filters: int) -> tf.Tensor:
    """Two sequential Conv2D -> BatchNorm -> LeakyReLU(0.1) operations."""
    for _ in range(2):
        x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.1)(x)
    return x


def up_block(x: tf.Tensor, filters: int) -> tf.Tensor:
    """Bilinear upsample x2, then 1x1 conv to halve channel count."""
    x = layers.UpSampling2D(size=2, interpolation='bilinear')(x)
    x = layers.Conv2D(filters, 1, padding='same')(x)
    return x


def build_n2v_unet(base_features: int = 32) -> keras.Model:
    """
    4-level encoder-decoder UNet for N2V, channels-last (NHWC).

    Encoder: 1 -> 32 -> 64 -> 128 -> 256  (MaxPool2D downsampling)
    Decoder: bilinear upsample + 1x1 proj + skip-concat + DoubleConvBlock
    Output:  Conv2D(1, 1) with no activation  (regression)

    Dynamic input shape (None, None, 1) allows reuse for both
    patch training (64x64) and tiled inference (256x256).
    """
    f      = base_features
    inputs = keras.Input(shape=(None, None, 1), name='noisy_input')

    # Encoder
    e1 = double_conv_block(inputs, f)                            # (B, H,   W,   32)
    e2 = double_conv_block(layers.MaxPool2D(2)(e1), f * 2)      # (B, H/2, W/2, 64)
    e3 = double_conv_block(layers.MaxPool2D(2)(e2), f * 4)      # (B, H/4, W/4, 128)
    e4 = double_conv_block(layers.MaxPool2D(2)(e3), f * 8)      # (B, H/8, W/8, 256)

    # Decoder with skip connections
    d3 = double_conv_block(layers.Concatenate()([up_block(e4, f * 4), e3]), f * 4)
    d2 = double_conv_block(layers.Concatenate()([up_block(d3, f * 2), e2]), f * 2)
    d1 = double_conv_block(layers.Concatenate()([up_block(d2, f),     e1]), f)

    # Output head — no activation (unclamped regression)
    outputs = layers.Conv2D(1, 1, name='denoised_output')(d1)

    return keras.Model(inputs, outputs, name='n2v_unet')


# ============================================================
# 4. Patch Generator and tf.data Pipeline
# ============================================================

def patch_generator(
    image:           np.ndarray,
    patch_size:      int,
    num_patches:     int,
    mask_ratio:      float = 0.006,
    neighbor_radius: int   = 5,
    seed:            int   = 42,
):
    """
    Python generator yielding (corrupted, target, mask) patch triples.
    Each output has shape (patch_size, patch_size, 1) float32 — channels-last.
    """
    H, W      = image.shape
    n_masked  = max(1, int(patch_size * patch_size * mask_ratio))
    rng       = np.random.default_rng(seed)

    for _ in range(num_patches):
        r0    = rng.integers(0, H - patch_size)
        c0    = rng.integers(0, W - patch_size)
        patch = image[r0:r0 + patch_size, c0:c0 + patch_size].copy()

        corrupted, mask = apply_n2v_masking(patch, n_masked, neighbor_radius, rng)

        # Channels-last: (P, P) -> (P, P, 1)
        yield (
            corrupted[..., np.newaxis],
            patch[..., np.newaxis],
            mask[..., np.newaxis],
        )


def make_dataset(
    image:       np.ndarray,
    patch_size:  int,
    num_patches: int,
    batch_size:  int,
    seed:        int,
) -> tf.data.Dataset:
    """Wrap patch_generator in a tf.data.Dataset with batching and prefetch."""
    P   = patch_size
    sig = (
        tf.TensorSpec(shape=(P, P, 1), dtype=tf.float32),   # corrupted input
        tf.TensorSpec(shape=(P, P, 1), dtype=tf.float32),   # clean target
        tf.TensorSpec(shape=(P, P, 1), dtype=tf.float32),   # mask
    )
    ds = tf.data.Dataset.from_generator(
        lambda: patch_generator(image, patch_size, num_patches, seed=seed),
        output_signature=sig,
    )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ============================================================
# 5. Training Loop
# ============================================================

def train_n2v(
    model:          keras.Model,
    image:          np.ndarray,
    patch_size:     int   = 64,
    batch_size:     int   = 128,
    num_epochs:     int   = 100,
    learning_rate:  float = 4e-4,
    val_percentage: float = 0.1,
) -> keras.Model:
    """
    Self-supervised N2V training on a single image.
    MSE loss computed only at masked pixel positions.
    """
    patches_per_epoch = 2000
    n_val   = max(1, int(patches_per_epoch * val_percentage))
    n_train = patches_per_epoch - n_val

    train_ds = make_dataset(image, patch_size, n_train, batch_size, seed=42)
    val_ds   = make_dataset(image, patch_size, n_val,   batch_size, seed=99)

    steps_per_epoch = max(1, n_train // batch_size)
    total_steps     = num_epochs * steps_per_epoch

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=learning_rate,
        decay_steps=total_steps,
        alpha=1e-6 / learning_rate,  # min_lr / initial_lr
    )
    optimizer = tf.keras.optimizers.Adam(lr_schedule)

    # Decorated with @tf.function for graph compilation (major speedup)
    @tf.function
    def train_step(
        noisy_in: tf.Tensor,
        clean_tgt: tf.Tensor,
        mask: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        with tf.GradientTape() as tape:
            pred = model(noisy_in, training=True)
            loss = tf.reduce_sum(tf.square((pred - clean_tgt) * mask))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, tf.reduce_sum(mask)

    @tf.function
    def val_step(
        noisy_in: tf.Tensor,
        clean_tgt: tf.Tensor,
        mask: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        pred = model(noisy_in, training=False)
        loss = tf.reduce_sum(tf.square((pred - clean_tgt) * mask))
        return loss, tf.reduce_sum(mask)

    n_params = model.count_params()
    print(f"Model parameters: {n_params:,}")
    print(f"Training: patch_size={patch_size}  batch_size={batch_size}  epochs={num_epochs}")
    print(f"Patches/epoch: train={n_train}  val={n_val}")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        # --- Train ---
        tr_loss, tr_count = 0.0, 0.0
        for noisy_in, clean_tgt, mask in train_ds:
            loss, n_m  = train_step(noisy_in, clean_tgt, mask)
            tr_loss   += float(loss)
            tr_count  += float(n_m)

        # --- Validate ---
        vl_loss, vl_count = 0.0, 0.0
        for noisy_in, clean_tgt, mask in val_ds:
            loss, n_m  = val_step(noisy_in, clean_tgt, mask)
            vl_loss   += float(loss)
            vl_count  += float(n_m)

        if epoch % 10 == 0 or epoch == 1:
            tr_mse = tr_loss / max(tr_count, 1)
            vl_mse = vl_loss / max(vl_count, 1)
            print(f"Epoch [{epoch:3d}/{num_epochs}]  "
                  f"train_loss={tr_mse:.6f}  val_loss={vl_mse:.6f}  "
                  f"elapsed={time.time() - t0:.1f}s")

    print("Training complete.")
    return model


# ============================================================
# 6. Tiled Inference with Hann-Window Blending
# ============================================================

def predict_tiled(
    model:        keras.Model,
    image:        np.ndarray,
    tile_size:    Tuple[int, int] = (256, 256),
    tile_overlap: Tuple[int, int] = (48, 48),
) -> np.ndarray:
    """
    Tiled inference with Hann-window blending to avoid seams.
    Returns denoised image as float32 (H, W).
    """
    H, W     = image.shape
    th, tw   = tile_size
    oh, ow   = tile_overlap
    stride_h = th - oh
    stride_w = tw - ow

    assert th <= H and tw <= W, \
        f"tile_size {tile_size} must be <= image size {image.shape}"
    assert th % 8 == 0 and tw % 8 == 0, \
        f"tile_size dimensions must be divisible by 8, got {tile_size}"

    # 2D Hann window — np.hanning is symmetric with zeros at endpoints,
    # equivalent to torch.hann_window(periodic=False)
    hann_2d = np.outer(np.hanning(th), np.hanning(tw)).astype(np.float32)

    output_sum = np.zeros((H, W), dtype=np.float64)
    weight_sum = np.zeros((H, W), dtype=np.float64)

    # Build tile origin lists, ensuring last tile reaches image edge
    row_starts = list(range(0, H - th + 1, stride_h))
    col_starts = list(range(0, W - tw + 1, stride_w))
    if row_starts[-1] + th < H:
        row_starts.append(H - th)
    if col_starts[-1] + tw < W:
        col_starts.append(W - tw)

    total_tiles = len(row_starts) * len(col_starts)
    processed   = 0

    for r0 in row_starts:
        for c0 in col_starts:
            tile_np = image[r0:r0 + th, c0:c0 + tw]
            # Channels-last: (1, th, tw, 1)
            tile_tf = tf.constant(tile_np[np.newaxis, ..., np.newaxis])

            pred_np = model(tile_tf, training=False).numpy().squeeze()  # (th, tw)

            output_sum[r0:r0 + th, c0:c0 + tw] += pred_np.astype(np.float64) * hann_2d
            weight_sum[r0:r0 + th, c0:c0 + tw] += hann_2d

            processed += 1
            if processed % 10 == 0 or processed == total_tiles:
                print(f"  Inference: {processed}/{total_tiles} tiles")

    denoised = (output_sum / np.maximum(weight_sum, 1e-8)).astype(np.float32)
    return denoised


# ============================================================
# 7. Save Outputs
# ============================================================

def save_outputs(
    image:    np.ndarray,
    denoised: np.ndarray,
    img_min:  float,
    img_max:  float,
    tif_path: str = "denoised_sem_tf.tif",
    png_path: str = "denoising_result.png",
) -> None:
    """Save denoised TIF (original value range) and side-by-side comparison PNG."""
    # Restore original grayscale range before saving
    denoised_original = (denoised * (img_max - img_min) + img_min).astype(np.float32)
    tifffile.imwrite(tif_path, denoised_original)
    print(f"Saved: {tif_path},  range: [{denoised_original.min():.3f}, {denoised_original.max():.3f}]")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image,    cmap='gray')
    axes[0].set_title('Original SEM Image')
    axes[0].axis('off')

    axes[1].imshow(denoised, cmap='gray')
    axes[1].set_title('N2V Denoised')
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
    image_path:   str             = "test_sem.tif",
    patch_size:   int             = 64,
    batch_size:   int             = 128,
    num_epochs:   int             = 100,
    tile_size:    Tuple[int, int] = (256, 256),
    tile_overlap: Tuple[int, int] = (48, 48),
) -> None:
    """Full N2V pipeline: load -> train -> predict -> save."""
    # Show GPU info
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU: {gpus[0].name}")
    else:
        print("No GPU found — running on CPU")

    # 1. Load image
    image, img_min, img_max = load_sem_image(image_path)
    print(f"Image shape: {image.shape},  range: [{img_min:.3f}, {img_max:.3f}]")

    # 2. Build model
    model = build_n2v_unet(base_features=32)

    # 3. Train
    model = train_n2v(
        model, image,
        patch_size=patch_size,
        batch_size=batch_size,
        num_epochs=num_epochs,
    )

    # 4. Tiled inference
    print("\nRunning tiled inference...")
    denoised = predict_tiled(
        model, image,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
    )

    # 5. Save outputs
    save_outputs(image, denoised, img_min, img_max)


if __name__ == '__main__':
    main()

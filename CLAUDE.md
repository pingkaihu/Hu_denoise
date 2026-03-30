# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **SEM (Scanning Electron Microscope) image denoising** project using self-supervised deep learning. The core approach is **Noise2Void (N2V)** via the CAREamics framework — no clean reference images needed, training works from a single noisy SEM image.

## Setup

```bash
pip install careamics tifffile matplotlib numpy bm3d
```

## Running the Denoiser

```bash
# Generate synthetic test image first (if no real SEM image available)
python test_sem.py

# Run the full N2V training + inference pipeline
python denoise.py
```

The script expects `test_sem.tif` (or another `.tif`/`.tiff`/`.png`) in the working directory and outputs:
- `denoised_sem.tif` — denoised result
- `denoising_result.png` — side-by-side comparison (original / denoised / difference ×3)

## Architecture

### Core Pipeline (`denoise.py`)

1. **Load** — `load_sem_image()` reads any `.tif`/`.tiff`/`.png`, converts RGB→grayscale if needed, normalizes to `[0, 1]` float32
2. **Configure** — `create_n2v_configuration()` sets patch size, batch size, epochs
3. **Train** — `CAREamist.train()` runs blind-spot self-supervised training on the image itself
4. **Predict** — `CAREamist.predict()` with tiled inference to avoid GPU OOM
5. **Output** — saves `.tif` and renders comparison figure

### Key Design Decisions

- **N2V works on a single image** — the network trains on patches extracted from the input image itself; no paired clean/noisy data required
- **Pixel-independent noise assumption** — N2V assumes noise is spatially independent per pixel (valid for SEM Poisson/Gaussian noise). For scan-line artifacts (horizontal/vertical stripes), use `struct_n2v_axis="horizontal"` or `"vertical"` in the config
- **Tiled prediction** — large images are processed in overlapping tiles (`tile_size`, `tile_overlap`) to avoid GPU memory issues

## Parameter Tuning Reference

| Situation | `patch_size` | `batch_size` | `num_epochs` |
|---|---|---|---|
| < 5 images, 8GB GPU | `[64, 64]` | `64` | `200` |
| 5–10 images, 8GB GPU | `[64, 64]` | `128` | `100` |
| High-res (> 2048px) | `[128, 128]` | `32` | `100` |
| CPU only | `[64, 64]` | `16` | `50` |

If inference hits OOM: reduce `tile_size` from `[256,256]` → `[128,128]` → `[64,64]`.

## Noise Type Decision

- **Uniform granular noise** (no stripes) → standard N2V (`denoise.py` as-is)
- **Horizontal/vertical scan stripes** → add `struct_n2v_axis="horizontal"` or `"vertical"` to `create_n2v_configuration()`
- **Unknown noise type** → run BM3D baseline first (`bm3d.bm3d(image, sigma_psd=0.05)`) to visually assess

See `guide.md` for full technical background on N2V, noise theory, and alternative methods.

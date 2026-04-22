# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SEM (Scanning Electron Microscope) image denoising using self-supervised deep learning. Core approach: **Noise2Void (N2V)** — no clean reference images needed, trains from a single noisy SEM image. Primary: pure PyTorch; legacy: CAREamics.

## Rules

1. **New script** — invoke the `denoise_new-script` skill before writing any denoising script; it enforces naming, pipeline, and output conventions.
2. **Output paths** — single-image: `data/denoised_sem_<tag>.tif`; multi-image: controlled by `--output_dir` flag.
3. **Script naming** — `denoise_<METHOD>.py` (single image), `denoise_<METHOD>_multi.py` (multiple images).
4. **Dependencies** — core: `torch tifffile matplotlib numpy`; add `scikit-learn` only for BIC scripts; no other new deps without explicit approval.
5. **Unknown noise type** — suggest running BM3D baseline first (`bm3d.bm3d(image, sigma_psd=0.05)`) before recommending a method.

## Setup

```bash
pip install torch tifffile matplotlib numpy
# Optional: pip install careamics bm3d  (legacy CAREamics + BM3D baseline)

# Generate synthetic test image (if no real SEM image available)
python test_sem.py

# Typical runs
python denoise_N2V_test.py                                                     # uniform noise, single image
python denoise_log_N2V.py                                                      # speckle / multiplicative
python denoise_N2V_multi.py --input_dir ./sem_images --output_dir ./denoised   # multi-image
```

Scripts read `.tif`/`.tiff`/`.png` from the working directory and write outputs under `data/`.

## Script Selection

### Uniform granular noise (no stripes)
- Single: `denoise_N2V_test.py` → `data/denoised_sem_test.tif`
- Multi: `denoise_N2V_multi.py`

### Speckle / multiplicative noise
- Single: `denoise_log_N2V.py` → `data/denoised_sem_log_torch.tif`
- Multi: `denoise_log_N2V_multi.py`

### Horizontal/vertical scan stripes
- `denoise_N2V_careamics.py` with `struct_n2v_axis="horizontal"` or `"vertical"`

### Mixed noise (Poisson + Gamma)
- **N2V + GMM noise model, auto BIC (recommended)** — `denoise_N2V_GMM_bic.py` (needs `scikit-learn`; uses PPN2V's parametric GMM but with scalar output; BIC auto-selects n_components) → `data/denoised_sem_PN2V_bic.tif`
  - Multi: `denoise_N2V_GMM_bic_multi.py`
- **N2V + GMM noise model, manual n_components** — `denoise_N2V_GMM.py` (same as above but you specify `--n_gaussians` directly; faster when n_components is known) → `data/denoised_sem_PN2V.tif`
  - Multi: `denoise_N2V_GMM_multi.py`
- **Paper-faithful PN2V** (non-parametric histogram 256×256, K=800 MMSE) — `denoise_PN2V_juglab.py` (`--calib_dir` for external calib; `--K` / `--n_bins`) → `data/denoised_sem_pn2v_juglab.tif`
  - Multi: `denoise_PN2V_juglab_multi.py`
- **Paper-faithful PPN2V** (parametric GMM + N2V bootstrap + K-sample MMSE posterior) — `denoise_PPN2V_juglab.py` (`--n2v_epochs`; `--calib_dir` to skip bootstrap; `--n_components`) → `data/denoised_sem_ppn2v_juglab.tif`
  - Multi: `denoise_PPN2V_juglab_multi.py`
- **Paper-faithful PPN2V + BIC auto-select** (same as above but BIC selects n_components automatically; needs `scikit-learn`) — `denoise_PPN2V_juglab_bic.py` (`--n_components 0` default=auto; `--bic_candidates`; `--bic_subsample`) → `data/denoised_sem_ppn2v_juglab_bic.tif`
  - Multi: `denoise_PPN2V_juglab_bic_multi.py`
- **Log + PPN2V** (log1p stabilises Gamma speckle; GMM models residual Poisson signal-dependency in log domain; for speckle-dominant + Poisson mixed noise) — `denoise_log_PPN2V_juglab.py` (`--n2v_epochs`; `--calib_dir`; `--n_components`) → `data/denoised_sem_log_ppn2v_juglab.tif`
  - Multi: `denoise_log_PPN2V_juglab_multi.py`

### Poisson/Gaussian additive noise (full receptive field, no blind-spot masking)
- Single: `denoise_GR2R.py` (Gaussian default; `--poisson --binomial_alpha 0.15` for Poisson; `--mc_samples`) → `data/denoised_sem_GR2R.tif`
- Multi: `denoise_GR2R_multi.py`

### Spatially correlated grain 2–4px
- Single: `denoise_apbsn_lee.py` (`--pd_stride 2`; `--save_model`/`--load_model`) → `data/denoised_sem_apbsn_lee.tif`
- Multi: `denoise_apbsn_lee_multi.py`

### Unknown noise distribution / N2V checkerboard artifacts
- `denoise_DIP.py` — no noise model assumption, EMA early stopping; ~3–5 min GPU → `data/denoised_sem_DIP.tif`

### Unified Gaussian/Poisson/Gamma (score-based)
- `denoise_N2Score.py` (`--noise_model gaussian/poisson/gamma`; `--blind` for auto σ via TV-norm) → `data/denoised_sem_N2Score.tif`

### Unknown noise type
- Run BM3D baseline first (`bm3d.bm3d(image, sigma_psd=0.05)`) to visually assess, then select above.

## Architecture

### Core Pipeline (`denoise_N2V_test.py` — optimized)

1. **Load** — `load_sem_image()` reads any `.tif`/`.tiff`/`.png`, converts RGB→grayscale, normalizes to `[0, 1]` float32; optional `use_log_transform` for multiplicative noise
2. **Split** — physical 80/20 spatial split into train/val regions (prevents data leakage from spatially correlated patches)
3. **Train** — blind-spot N2V training with vectorized numpy masking (no Python loops in `__getitem__`)
4. **Predict** — batched tiled inference with Hann-window blending and reflection padding for edge/small-image safety
5. **Output** — saves `.tif` and renders side-by-side comparison figure

### Key Design Decisions

- **N2V works on a single image** — patches extracted from the input itself; no paired clean/noisy data required
- **Pixel-independent noise assumption** — N2V assumes spatially independent per-pixel noise (valid for SEM Poisson/Gaussian). For scan-line artifacts, use `denoise_N2V_careamics.py` with `struct_n2v_axis="horizontal"` or `"vertical"`
- **Log transform** — `denoise_log_N2V.py` uses `log1p` to convert multiplicative speckle to additive AWGN before training, then `expm1` to invert after prediction
- **Tiled prediction** — Hann-window weighting blends tile seams; reflection padding handles images smaller than `tile_size` or non-multiples of 8 (UNet requirement)

## Parameter Tuning

| Situation | `patch_size` | `batch_size` | `num_epochs` |
|---|---|---|---|
| Single image, 8GB GPU | `[64, 64]` | `64` | `200` |
| Multiple images, 8GB GPU | `[64, 64]` | `128` | `100` |
| High-res (> 2048px) | `[128, 128]` | `32` | `100` |
| CPU only | `[64, 64]` | `16` | `50` |

OOM during inference: reduce `tile_size` from `[256,256]` → `[128,128]` → `[64,64]`.

## Documentation

- `document/guide.md` — full technical background on N2V, noise theory, and alternative methods
- `document/N2V_optimization.md` — detailed analysis of the five optimizations in `denoise_N2V_test.py`
- `document/speckle_denoising_strategy.md` — speckle and shot noise processing strategies

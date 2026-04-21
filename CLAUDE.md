# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **SEM (Scanning Electron Microscope) image denoising** project using self-supervised deep learning. The core approach is **Noise2Void (N2V)** — no clean reference images needed, training works from a single noisy SEM image. Implementations use pure PyTorch (primary) and CAREamics (legacy).

## Setup

```bash
pip install torch tifffile matplotlib numpy
# Optional: pip install careamics bm3d  (for legacy CAREamics script and BM3D baseline)
```

## Script Selection Guide

| Script | When to use | Output |
|---|---|---|
| `denoise_N2V.py` | Standard single-image N2V, baseline PyTorch | `data/denoised_sem_N2V.tif` |
| `denoise_N2V_test.py` | **Recommended for uniform noise** — optimized version (vectorized masking, batched inference, physical train/val split, edge padding) | `data/denoised_sem_test.tif` |
| `denoise_PN2V.py` | **Recommended for mixed noise** — PN2V pure PyTorch; GMM models raw Poisson-Gamma directly; no GAT pre-processing; includes low-count diagnostic. Note: uses parametric GMM (official uses non-parametric histogram); single scalar output (official uses K=800 samples) | `data/denoised_sem_PN2V.tif` |
| `denoise_PN2V_bic.py` | **Mixed noise, auto GMM capacity** — same as PN2V but runs BIC (Schwarz 1978) over K ∈ {2,3,5,7} before training; recommended when ENL < 3 (strong speckle); requires `scikit-learn` | `data/denoised_sem_PN2V_bic.tif` |
| `denoise_PN2V_juglab.py` | **PN2V paper-faithful (juglab port)** — non-parametric 2D histogram noise model (256×256 bins); K=800 output samples; MMSE posterior mean inference `Σ p(y\|s_k)·s_k / Σ p(y\|s_k)`; self-calibration by default or `--calib_dir` for external images; `--K` / `--n_bins` / `--batch_size` | `data/denoised_sem_pn2v_juglab.tif` |
| `denoise_PN2V_juglab_multi.py` | **PN2V juglab, multiple images** — same histogram + K=800 MMSE as juglab.py; histogram built from ALL training images pooled; shared UNet; `--train_dir` / `--save_model` / `--load_model`; per-image MMSE + prior outputs | `--output_dir` flag |
| `denoise_PPN2V_juglab.py` | **PPN2V paper-faithful (juglab port)** — parametric signal-dependent GMM noise model (`p(y\|s)=Σ α_k(s)·N`); bootstrap: runs N2V for `--n2v_epochs` → pseudo-clean → fit GMM → freeze → train K=800 UNet; or `--calib_dir` to skip bootstrap; `--n_components` / `--gmm_steps` | `data/denoised_sem_ppn2v_juglab.tif` |
| `denoise_PPN2V_juglab_multi.py` | **PPN2V juglab, multiple images** — same parametric GMM + bootstrap as PPN2V_juglab.py; shared N2V bootstraps on ALL images; GMM fitted from pooled pseudo-clean pairs; shared PN2V UNet; `--train_dir` / `--save_model` / `--load_model` | `--output_dir` flag |
| `denoise_log_N2V.py` | Speckle / multiplicative noise — applies log transform before training | `data/denoised_sem_log_torch.tif` |
| `denoise_N2V_multi.py` | Multiple images under similar conditions — one shared N2V model (MSE loss) | `--output_dir` flag |
| `denoise_log_N2V_multi.py` | **Multiple images, speckle/multiplicative noise** — log-domain shared N2V; per-image low-count floor diagnostic; same CLI as N2V_multi | `--output_dir` flag |
| `denoise_PN2V_multi.py` | **Multiple images, mixed noise** — shared UNet + shared GMM; pools pixel pairs from all images for richer noise statistics; same CLI as N2V_multi | `--output_dir` flag |
| `denoise_PN2V_bic_multi.py` | **Multiple images, mixed noise, auto GMM capacity** — same as PN2V_multi but BIC evaluated on pooled pixel pairs from all images; `--bic_candidates` / `--bic_subsample`; requires `scikit-learn` | `--output_dir` flag |
| `denoise_GR2R_multi.py` | **Multiple images, unknown additive / shot noise** — R2R shared-ε pair (Gaussian) or GR2R Binomial splitting (Poisson, `--poisson --binomial_alpha 0.15`); per-image σ auto-estimated; `--mc_samples` for MC inference averaging; `--save_model`/`--load_model` | `--output_dir` flag |
| `denoise_apbsn.py` | AP-BSN (CVPR 2022) — real-world noise, asymmetric PD + blind-spot | configurable |
| `denoise_apbsn_faithful.py` | **AP-BSN paper-faithful** — DBSNl architecture (CentralMaskedConv2d, dilated branches), L1 loss on all pixels, asymmetric PD (pd_a train ≠ pd_b infer), R3 refinement | `data/denoised_sem_apbsn_faithful.tif` |
| `denoise_apbsn_faithful_multi.py` | **AP-BSN faithful, multiple images** — same DBSNl + R3; trains one shared model on all images; `--train_dir` / `--save_model` / `--load_model` | `--output_dir` flag |
| `denoise_apbsn_lee.py` | **AP-BSN official-style port** — closest to wooseoklee4/AP-BSN repo; APBSN wrapper (PD inside model.forward), pixel_shuffle_down/up_sampling (official util.py), official R3 (bsn() direct, no PD in refinement passes); `--save_model`/`--load_model` | `data/denoised_sem_apbsn_lee.tif` |
| `denoise_apbsn_lee_multi.py` | **AP-BSN official-style, multiple images** — same APBSN+DBSNl as lee.py; raw-crop dataset (no PD pre-computation); shared model trained on image pool; per-image `APBSN.denoise()` with torch R3; `--train_dir` / `--save_model` / `--load_model` | `--output_dir` flag |
| `denoise_DIP.py` | Deep Image Prior (CVPR 2018) — no dataset, single-image generator, no noise model assumption, EMA early stopping; ~3-5 min on GPU | `data/denoised_sem_DIP.tif` |
| `denoise_GR2R.py` | **R2R/GR2R** — no blind-spot masking; Gaussian: R2R shared-ε pair (Pang et al. CVPR 2021); Poisson shot noise: GR2R Binomial splitting (`--poisson --binomial_alpha 0.15`, Monroy et al. CVPR 2025); `--mc_samples` for MC inference averaging; auto-estimates noise std | `data/denoised_sem_GR2R.tif` |
| `denoise_N2Score.py` | **Noise2Score (NeurIPS 2021)** — AR-DAE score estimation + Tweedie's formula; supports Gaussian (`--noise_model gaussian`), Poisson (`--noise_model poisson --poisson_zeta 0.05`), Gamma; `--blind` for automatic σ search via TV-norm | `data/denoised_sem_N2Score.tif` |
| `denoise_N2V_careamics.py` | CAREamics-based pipeline (legacy) | `denoised_sem.tif` |

## Running the Denoiser

```bash
# Generate synthetic test image first (if no real SEM image available)
python test_sem.py

# Recommended: run the optimized N2V pipeline
python denoise_N2V_test.py

# Multi-image denoising
python denoise_N2V_multi.py --input_dir ./sem_images --output_dir ./denoised

# Multiplicative / speckle noise
python denoise_log_N2V.py
```

Scripts read `.tif`/`.tiff`/`.png` from the working directory and write outputs under `data/`.

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

## Parameter Tuning Reference

| Situation | `patch_size` | `batch_size` | `num_epochs` |
|---|---|---|---|
| Single image, 8GB GPU | `[64, 64]` | `64` | `200` |
| Multiple images, 8GB GPU | `[64, 64]` | `128` | `100` |
| High-res (> 2048px) | `[128, 128]` | `32` | `100` |
| CPU only | `[64, 64]` | `16` | `50` |

If inference hits OOM: reduce `tile_size` from `[256,256]` → `[128,128]` → `[64,64]`.

## Noise Type Decision

- **Uniform granular noise** (no stripes) → `denoise_N2V_test.py`
- **Speckle / multiplicative noise** → `denoise_log_N2V.py`
- **Multiple images, speckle/multiplicative** → `denoise_log_N2V_multi.py`
- **Horizontal/vertical scan stripes** → `denoise_N2V_careamics.py` with `struct_n2v_axis`
- **Multiple images, same conditions** → `denoise_N2V_multi.py`
- **Single image, Poisson/Gaussian additive noise (full receptive field)** → `denoise_GR2R.py`
- **Multiple images, unknown additive / Poisson shot noise** → `denoise_GR2R_multi.py`
- **Mixed noise, Poisson + Gamma, auto GMM capacity** → `denoise_PN2V_bic.py` (BIC selects K; needs `scikit-learn`)
- **Same as above, multiple images** → `denoise_PN2V_bic_multi.py`
- **Mixed noise, paper-faithful PN2V (histogram + K=800 MMSE)** → `denoise_PN2V_juglab.py` (juglab/pn2v port; self-calibration default; `--calib_dir` for external images)
- **Same as above, multiple images** → `denoise_PN2V_juglab_multi.py` (shared histogram from pooled images; `--train_dir` / `--save_model`)
- **Mixed noise, paper-faithful PPN2V (parametric GMM + bootstrap)** → `denoise_PPN2V_juglab.py` (signal-dependent GMM; bootstrap N2V auto-calibrates without external images; `--calib_dir` to use external images; `--n_components` / `--n2v_epochs`)
- **Same as above, multiple images** → `denoise_PPN2V_juglab_multi.py` (shared bootstrap N2V on all images; pooled GMM fitting; `--train_dir` / `--save_model`)
- **Mixed noise, spatially correlated grain 2–4px** → `denoise_apbsn_lee.py` (`--pd_stride 2`; reusable model)
- **Same as above, multiple images** → `denoise_apbsn_lee_multi.py`
- **Real-world complex noise** → `denoise_apbsn.py`
- **Real-world complex noise, paper-faithful DBSNl + R3** → `denoise_apbsn_faithful.py`
- **Same as above, multiple images** → `denoise_apbsn_faithful_multi.py`
- **Unknown noise distribution** → `denoise_DIP.py` (no noise model assumption)
- **N2V leaves checkerboard artifacts** → `denoise_DIP.py`
- **Unified Gaussian/Poisson/Gamma, score-based framework** → `denoise_N2Score.py` (Tweedie's formula; `--blind` for auto σ; NeurIPS 2021)
- **Unknown noise type** → run BM3D baseline first (`bm3d.bm3d(image, sigma_psd=0.05)`) to visually assess

## Documentation

- `document/guide.md` — full technical background on N2V, noise theory, and alternative methods
- `document/N2V_optimization.md` — detailed analysis of the five optimizations in `denoise_N2V_test.py`
- `document/speckle_denoising_strategy.md` — speckle and shot noise processing strategies

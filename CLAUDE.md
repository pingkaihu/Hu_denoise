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

## Project Structure

```
Hu_denoise/
├── data/                  # input images + denoised outputs (.tif, .png)
├── document/              # technical guides (.md)
├── reference/             # academic papers (PDF)
├── denoise_*.py           # denoising scripts (single image)
├── denoise_*_multi.py     # denoising scripts (multiple images)
└── test_sem.py / test_gen_multi.py / convert_to_tif.py   # utilities
```

## Reference

See [README.md](README.md) for script selection by noise type, setup instructions, parameter tuning, and documentation links.

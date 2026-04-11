---
name: denoise-new-script
description: >
  Generate a new SEM image denoising Python script for this project, following
  the established conventions (UNet, tiled inference, side-by-side output).
  Use this skill whenever the user wants to implement a new denoising algorithm,
  try a new method, or says things like "幫我建一個 X 的去噪腳本", "implement X
  denoising", "新增 X 方法", "I want to try X algorithm", "add support for X
  noise model", "create a script for Y method" — even if they don't say "script"
  explicitly. Also trigger when discussing a new paper or technique the user
  wants to test in this SEM context.
---

# denoise-new-script

You are generating a new Python script for the SEM image denoising project in
the current working directory. Every script must feel consistent with the
existing ones — a developer should be able to open any two scripts side by side
and immediately see they belong to the same codebase.

## Before writing

1. Ask the user (or infer from context) two things:
   - **Algorithm name / paper** — e.g., "Noise2Self", "Recorrupted-to-Recorrupted", "HDN"
   - **What differs from standard N2V** — new loss? different masking? noise model?

   If the user already gave enough context, skip the question.

2. Read `denoise_N2V_multi.py` briefly to confirm you have the current shared
   boilerplate (it is the most complete template). You've likely already seen it
   in the conversation, so re-reading is optional.

3. Consult the reference documents below as needed — they contain decisions
   already made in this project that the new script should respect:

   | Document | When to read it |
   |---|---|
   | `document/N2V_optimization.md` | **Always** — read sections 1, 2, 4 only; these three are the ones every script should inherit (vectorized masking, batched inference, reflection padding). Skip section 3 (physical train/val split — evaluated and not recommended for general use) and section 5 (log transform — only applies to the dedicated log-variant script). |
   | `document/guide.md` | When you need theoretical background on the algorithm the user requested |
   | `document/settings_guide.md` | When setting parameter defaults — this project runs on RTX 3080 (10 GB), so use those as the baseline |
   | `document/critique.md` | When the algorithm involves GAT, PN2V, or mixed-noise models — read to avoid known pitfalls (e.g., GAT+PN2V redundancy) |
   | `document/Mixed_Noise_Speckle_ShotNoise_Guide.md` | When the target noise is mixed (Poisson + Gaussian) or speckle |

---

## Script structure

Every script follows these numbered sections, in order:

```
# ============================================================
# SEM Image Denoising — <Algorithm Name> (pure PyTorch)
# ============================================================
# Based on: <paper citation if applicable>
#
# Differences from denoise_N2V.py:
#   + <what's new>
#
# Identical to denoise_N2V.py:
#   = <what's reused verbatim>
#
# Requirements: torch>=2.0.0  tifffile  matplotlib  numpy
# Usage:
#   python <filename>.py
#   python <filename>.py --<key_arg> <value>
# ============================================================

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# ... rest of imports ...

torch.set_float32_matmul_precision('high')

# 1. Image Loading     ← always identical to other scripts
# 2. UNet Architecture ← always identical (DoubleConvBlock + N2VUNet)
# 3. <Algorithm-specific components>  ← the novel part
# 4. Dataset           ← modify if masking strategy changes
# 5. Training Loop     ← modify loss function / optimizer if needed
# 6. Tiled Inference   ← always identical (_compute_padding + predict_tiled)
# 7. Save Outputs      ← always identical (3-panel: Original | Denoised | Diff×3)
# 8. Main Pipeline     ← wire everything together; use argparse for CLI
```

---

## What stays the same across all scripts

Copy these verbatim (they are battle-tested and should not diverge):

**`load_sem_image(path)`** — reads `.tif`/`.tiff`/`.png`, converts RGB→gray
with ITU weights `[0.2989, 0.5870, 0.1140]`, normalises to float32 `[0,1]`,
returns `(image, img_min, img_max)`.

**`DoubleConvBlock`** — two Conv2d+BN+LeakyReLU(0.1) layers.

**`N2VUNet`** — 4-level encoder-decoder, `base_features=32`, input must be
divisible by 8.

**`_compute_padding` + `predict_tiled`** — reflection padding per-axis
independently (Opt 4), Hann-window blending, batched GPU inference with
`infer_batch_size=8` (Opt 2). Use the version from `denoise_N2V_multi.py`
which has the per-axis padding fix (not the simpler version in `denoise_N2V.py`).

**`save_outputs`** — saves `.tif` in original value range, then a 3-panel PNG:
Original | Denoised | `abs(diff) * 3` on `'hot'` colormap.

**Thread env vars + `torch.set_float32_matmul_precision('high')`** — always at
the top.

---

## What changes per algorithm

The novel part lives between sections 3–5. Examples:

| Algorithm | Change |
|---|---|
| N2V standard | MSE loss on masked pixels only |
| PN2V | Add `GMMNoiseModel`; pretrain GMM; replace MSE with NLL |
| Log-N2V | `log1p` on load, `expm1` on output; rest unchanged |
| Noise2Self | J-invariant masking (each pixel predicts itself via partitioned subsets) |
| HDN | Hierarchical VAE encoder; KL + reconstruction loss |
| Recorrupted-to-Recorrupted (R2R) | Re-corrupt input twice; loss on both outputs |

Document the difference clearly in the file header's "Differences from
denoise_N2V.py" block.

---

## Output naming

Output files go under `data/`:
- TIF: `data/denoised_sem_<ALGORITHM>.tif`
- PNG: `data/denoising_result_<ALGORITHM>.png`

Use ALL_CAPS short name for `<ALGORITHM>`, e.g., `N2S`, `HDN`, `R2R`, `LOG`.

Update `CLAUDE.md`'s Script Selection Guide table after writing the script —
add a row with the filename, when to use it, and the output path.

---

## CLI interface

Use `argparse` in `main()`. Minimum args:
```python
--input   # path to input .tif (default: 'data/test_sem.tif')
--output  # path to output .tif (default: auto-set per algorithm)
--epochs  # int (default: 100)
--patch_size  # int (default: 64)
--batch_size  # int (default: 128)
```

Add algorithm-specific args as needed (e.g., `--n_gaussians` for PN2V).

---

## Quality checklist before finishing

- [ ] `patch_size % 8 == 0` asserted in Dataset
- [ ] `data/` directory created with `os.makedirs("data", exist_ok=True)`
- [ ] `os.environ` thread vars set before any import that might fork
- [ ] File header lists paper, differences, and identical sections
- [ ] CLAUDE.md Script Selection Guide updated with new row

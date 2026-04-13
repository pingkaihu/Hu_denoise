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
   | `document/guide.md` | When you need theoretical background on the algorithm the user requested |
   | `document/settings_guide.md` | When setting parameter defaults — this project runs on RTX 3080 (10 GB), so use those as the baseline |
   | `document/Mixed_Noise_Speckle_ShotNoise_Guide.md` | When the target noise is mixed (Poisson + Gaussian) or speckle |

   Implementation patterns (vectorized masking, batched inference, reflection padding) and known
   pitfalls (GAT+PN2V redundancy, low-count instability, blind-spot oversmoothing, batch drift)
   are documented inline in the sections below — no external document read needed.

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

**`Dataset.__getitem__` masking** — vectorized numpy, no Python loops.
This is the primary DataLoader CPU bottleneck; never use a `for` loop over
masked pixels:
```python
dr = self.rng.integers(-rad, rad + 1, size=self.n_masked)
dc = self.rng.integers(-rad, rad + 1, size=self.n_masked)
zero_mask = (dr == 0) & (dc == 0)
if np.any(zero_mask):
    dr[zero_mask] += self.rng.choice([-1, 1], size=int(zero_mask.sum()))
nr = np.clip(rows + dr, 0, P - 1)
nc = np.clip(cols + dc, 0, P - 1)
corrupted[rows, cols] = patch[nr, nc]
mask[rows, cols] = 1.0
```

**`_compute_padding` + `predict_tiled`** — reflection padding **per-axis
independently** (not a single shared max — that is the bug in `denoise_N2V.py`),
Hann-window blending, batched GPU inference with `infer_batch_size=8`.
Key padding pattern:
```python
pad_h = max(0, tile_h - H, (8 - H % 8) % 8)
pad_w = max(0, tile_w - W, (8 - W % 8) % 8)
padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
# ... inference on padded ...
denoised = denoised_pad[:H, :W]   # crop back to original size
```
Use the version from `denoise_N2V_multi.py`.

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
| Recorrupted-to-Recorrupted (R2R) / GR2R | Re-corrupt input twice with Gaussian or Poisson noise; MSE on all pixels (no masking); auto-estimate noise std via Laplacian MAD — see `denoise_GR2R.py` |
| AP-BSN | Asymmetric pixel-shuffle downsampling + blind-spot network; no noise model required; separate PD stride for train vs. inference — see `denoise_apbsn.py` |

Document the difference clearly in the file header's "Differences from
denoise_N2V.py" block.

---

## Known Pitfalls

Decisions and failure modes already investigated in this project — don't repeat them.

**1. Never combine GAT + PN2V**  
GAT converts Poisson-Gaussian noise into AWGN. If it works correctly, standard N2V with MSE
is already the mathematical optimum — PN2V adds nothing. Worse, the GMM will overfit GAT's
low-signal artefacts and treat them as noise structure.  
Rule: GAT → use standard N2V. Skipping GAT → use PN2V on raw pixel values.

**2. Log / Anscombe transforms are unstable at low counts**  
When many pixels are near zero (dark SEM backgrounds, low-dose images), `log1p` and `sqrt`
amplify noise non-linearly. The inverse transform produces hazy grey backgrounds or false
structure. Before applying any such transform, run a diagnostic:
```python
low_frac = np.mean(image < 0.05)   # in normalised [0,1] space
if low_frac > 0.10:
    # clip a floor or switch to PN2V / DIP instead
    image = np.maximum(image, 0.01)
```

**3. Expanding the blind-spot radius causes oversmoothing**  
A larger masked region forces predictions from 2–3 pixels away. For SEM images where fine
structures (nanowires, grain boundaries) can span 1–2 pixels, this irrecoverably smears detail.
Keep the default `rad = 2` (neighbourhood size). Do not increase it to handle spatially-
correlated speckle — use AP-BSN or DIP instead.

**4. Multi-image scripts: noise level drift**  
A model trained on image 1 may hallucinate on image 50 if SEM charging, beam current drift,
or slight defocus shifted the noise floor. For `_multi` scripts applied to long acquisition
series: support an optional `--histogram_match` flag that normalises each image against the
training set's intensity distribution before inference.

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
--input       # path to input .tif (default: 'data/test_sem.tif')
--output      # path to output .tif (default: auto-set per algorithm)
--epochs      # int (default: 100)
--patch_size  # int (default: 64)
--batch_size  # int (default: 128)
--tile_size   # int (default: 256) — reduce to 128 or 64 if inference hits OOM
--device      # str (default: 'cuda' if available else 'cpu')
```

Add algorithm-specific args as needed (e.g., `--n_gaussians` for PN2V).

---

## Quality checklist before finishing

- [ ] `patch_size % 8 == 0` asserted in Dataset
- [ ] `data/` directory created with `os.makedirs("data", exist_ok=True)`
- [ ] `os.environ` thread vars set before any import that might fork
- [ ] File header lists paper, differences, and identical sections
- [ ] `--tile_size` and `--device` args present in argparse
- [ ] CLAUDE.md **Script Selection Guide** table updated with new row
- [ ] CLAUDE.md **Noise Type Decision** section updated with new algorithm entry
- [ ] Script runs end-to-end with defaults: `python <script>.py` succeeds on `data/test_sem.tif`

---

## Early stopping

For algorithms that overfit without a stopping criterion (generative models, deep priors, VAE-based):

- Track a smoothed validation loss with EMA: `ema = 0.9 * ema + 0.1 * val_loss`
- Stop when EMA stops improving for N patience steps
- Reference implementation: `denoise_DIP.py`

Standard discriminative scripts (N2V variants, GR2R, AP-BSN) do **not** need early stopping — use fixed epochs.

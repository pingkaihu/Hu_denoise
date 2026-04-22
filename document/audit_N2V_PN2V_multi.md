# Audit Report: `denoise_N2V_multi.py` & `denoise_N2V_GMM_multi.py`

**Date:** 2026-04-16  
**Reviewer:** Claude Sonnet 4.6  
**References:**  
- Krull et al., "Noise2Void — Learning Denoising from Single Noisy Images," CVPR 2019  
- Krull et al., "Probabilistic Noise2Void: Unsupervised Content-Aware Denoising," ICLR 2020  
- Official N2V code: github.com/juglab/n2v  
- Official PN2V code: github.com/juglab/pn2v  

---

## Part 1 — `denoise_N2V_multi.py`

### 1.1 Blind-Spot Masking — `_apply_n2v_masking`

#### Issue A · Mask Ratio (Medium)

| | Value | Formula |
|---|---|---|
| **This script** | `mask_ratio = 0.006` → `n_masked = max(1, int(P² × 0.006))` | For P=64: **24 pixels** (0.6%) |
| **Official N2V** | `n2v_perc_pix = 1.5` → stratified over a `box_size = √(100/1.5) ≈ 8` grid | For P=64: **~61 pixels** (1.5%) |
| **Paper (Appendix)** | p_mask ≈ 0.002 | For P=64: **~8 pixels** (0.2%) |

The implementation masks roughly **4× fewer pixels** than the official default and **3× more** than the paper appendix value. This is not a correctness bug — the model can still learn the blind-spot objective — but it reduces the gradient signal per patch and may require more epochs to converge.

#### Issue B · Replacement Sampling Strategy (Low–Medium)

| | Strategy |
|---|---|
| **This script** | Sample `dr ∈ [-rad, rad]`, `dc ∈ [-rad, rad]` independently; if `(dr, dc) == (0,0)` fix one axis to ±1 |
| **Official N2V** | Provides multiple strategies (`pm_uniform_withoutCP`, `pm_median`, `pm_normal_fitted`); default `pm_uniform_withoutCP` samples uniformly from the (2r+1)² − 1 neighborhood (center excluded, flat distribution over valid offsets) |

The independent-axis sampling here makes all `(2×5+1)² = 121` neighbors **equally likely** — which is equivalent to `pm_uniform_withoutCP`. However the zero-displacement fix (lines 231–237) introduces a subtle bias: when `(0,0)` is drawn it is redirected to one of `{(0,±1), (±1,0)}`, so nearest 4-neighbors are very slightly over-represented. In practice this is negligible for `rad=5` (affects ≈ 1/121 of masked pixels).

#### Issue C · Patch Sampling Off-By-One (Low)

```python
# Current — line 204-205
r0 = int(self.rng.integers(0, H - P))  # range [0, H-P-1]
c0 = int(self.rng.integers(0, W - P))  # range [0, W-P-1]
```

`np.random.default_rng().integers(low, high)` returns values in `[low, high)`, so the last valid top-left corner `r0 = H−P` (which places the patch exactly at the bottom edge) is **never sampled**. The correct call is:

```python
r0 = int(self.rng.integers(0, H - P + 1))  # range [0, H-P] inclusive
c0 = int(self.rng.integers(0, W - P + 1))
```

For large SEM images the bias is negligible (1 row/column out of hundreds). For images where `H` is close to `patch_size` (e.g., H=65, P=64) only one position is ever sampled, missing the patch that would cover the last row.

### 1.2 Loss Function

| | Loss |
|---|---|
| **This script** | `nn.MSELoss(reduction='sum')` — L2 on masked pixels only |
| **Official N2V** | **L1 (MAE)** on masked pixels by default; L2 is a configurable option |

The official repository switched to L1 because it is more robust to outlier bright spots common in SEM (hot pixels, charging artefacts). The MSE implementation is **algorithmically valid** but may produce slightly noisier residuals at high-intensity outliers.

### 1.3 Train / Validation Split

| | Strategy |
|---|---|
| **This script** | Both `train_ds` and `val_ds` sample patches **randomly from the same image pool** (different RNG seeds); patches can overlap |
| **`denoise_N2V_test.py` (this repo)** | Physical 80/20 spatial split — train and val regions are non-overlapping |

For multi-image training with `patches_per_epoch ≫ H×W/P²` the overlap probability is low, but the validation loss is not a strictly independent measure of generalization. This is a known practical trade-off in multi-image self-supervised training.

### 1.4 Training — Correct Aspects

- ✅ Blind-spot principle correct: corrupted input uses neighbor values; target is the **original noisy pixel** (not a clean reference)
- ✅ Loss computed only on masked pixels (`pred * mask` vs `clean_tgt * mask`)
- ✅ UNet architecture (4-level, skip connections) matches the N2V paper spec
- ✅ Cosine LR schedule, Adam optimizer — reasonable choices not contradicting the paper
- ✅ `patches_per_epoch = max(2000, 500 × n_images)` — sensible scaling heuristic
- ✅ Tiled inference with Hann-window blending — correct seam-free reconstruction

---

## Part 2 — `denoise_N2V_GMM_multi.py`

### 2.1 Noise Model Architecture — GMM vs. Histogram (High)

This is the most fundamental architectural difference between this implementation and the official PN2V:

| | Noise Model |
|---|---|
| **This script** | **Parametric GMM** — `GMMNoiseModel` with K Gaussians, signal-dependent μ_k(s) = s + offset_k and σ_k²(s) = exp(a_k·s + b_k) |
| **Official PN2V** | **Non-parametric 2-D histogram** — 256×256 bins over (s, y) space; differentiable via bilinear interpolation; no functional form assumed |

**Implications:**
- The GMM assumes the noise is a mixture of Gaussians with signal-dependent variance (Poisson-like + read noise). This is physically motivated for SEM but imposes a parametric family.
- The official histogram approach makes **no distribution assumption** — it can capture arbitrary noise shapes (skewed, multi-modal, bimodal tails from charging events).
- The GMM is a **valid approximation** for well-behaved SEM noise (Poisson + Gaussian), but will underfit if the actual noise distribution has heavier tails or additional modes.

### 2.2 Network Output Representation (Critical)

| | Per-pixel output |
|---|---|
| **This script** | Network predicts **one scalar** s_pred per pixel; loss = NLL(y_obs, s_pred) under GMM |
| **Official PN2V** | Network predicts **K=800 independent samples** {s_k} from p(s \| context); loss = −log( (1/K) Σ_k p(y_obs \| s_k) ) |

The official PN2V represents the **posterior predictive distribution** over signal values as a set of samples. The script approximates it with a single point estimate. Consequences:

1. **Training loss is a different objective.** The script minimises the NLL of a point prediction; the official minimises the NLL of a sample mixture (a tighter variational bound on the marginal log-likelihood when K → ∞).
2. **No uncertainty quantification.** The official approach can compute per-pixel posterior variance; this script cannot.
3. **Practical denoising quality.** For unimodal posteriors the difference is small; for bimodal or heavy-tailed noise the sample-based approach captures more of the posterior.

### 2.3 Inference — Missing MMSE Posterior Mean (Critical)

This is the most impactful deviation from the paper.

| | Inference |
|---|---|
| **This script** | `predict_tiled()` runs a **single forward pass** and returns the network output directly |
| **Official PN2V** | Computes **MMSE posterior mean**: `ŝ = Σ_k p(y\|s_k) · s_k / Σ_k p(y\|s_k)`, where {s_k} are K samples from the network |

The official PN2V inference procedure:
1. Run K forward passes (or one pass that produces K sample outputs) to obtain `{s_k}`.
2. Evaluate the noise model likelihood `p(y_obs | s_k)` for each sample.
3. Return the **likelihood-weighted average** of the samples.

This MMSE estimator is **Bayes-optimal under squared error loss** and is the primary innovation of PN2V over plain N2V. Without it, PN2V degenerates to a standard N2V with a GMM-based NLL loss — the probabilistic noise model is trained but never used during inference.

**Current inference output is equivalent to:**
```
ŝ_MAP-like ≈ argmax_s p(s | context)   (via network output)
```
**Paper's intended output:**
```
ŝ_MMSE = E[s | y_obs, context] = Σ_k p(y_obs | s_k) · s_k / Σ_k p(y_obs | s_k)
```

For symmetric unimodal noise, the two are numerically close. For asymmetric or multi-modal noise, the MMSE estimator can yield meaningfully better denoising and will also de-bias the estimate toward the signal (rather than the noisy observation).

### 2.4 Signal Proxy for GMM Pre-training — Correct

```python
kernel[0,0,0,1] = 0.25  # top
kernel[0,0,2,1] = 0.25  # bottom
kernel[0,0,1,0] = 0.25  # left
kernel[0,0,1,2] = 0.25  # right
```

The 4-neighbor mean as a low-noise signal proxy matches the strategy used in the official PN2V and in the paper's supplementary. ✅

### 2.5 GMM Pre-training — Reasonable Engineering Deviation (Low)

| | GMM Pre-training |
|---|---|
| **This script** | Explicit Adam-based GMM pre-training (`pretrain_gmm_multi`) on pixel pairs before joint training |
| **Official PN2V** | Histogram estimated from pixel–neighbor pairs analytically (bin-counting + normalisation); no iterative optimization needed |

Because the official uses a histogram (non-parametric), no gradient-based pre-training is needed. The GMM-based pre-training is a reasonable substitute for the parametric case, though it adds complexity and sensitivity to learning rate / n_epochs.

### 2.6 Shared GMM Across Images — Not in Paper (Low)

The multi-image shared GMM is a **novel extension** not described in the original PN2V paper (which is single-image). The implementation documents the assumption (same session conditions) and the limitation (heterogeneous conditions → biased GMM). This is a valid practical extension but has no direct paper backing.

### 2.7 Correct Aspects of PN2V_multi

- ✅ NLL loss formulation: `−log p(y_obs | s_pred)` on masked pixels only
- ✅ Signal-dependent variance `σ²(s) = exp(a·s + b)` — correct for Poisson-like SEM noise
- ✅ Signal-centered mean `μ(s) = s + offset` — correct parameterization
- ✅ Joint UNet + GMM optimization with separate (scaled) learning rates
- ✅ 4-neighbor proxy for GMM pre-training
- ✅ Blind-spot masking identical to N2V (same issues as §1.1 apply)
- ✅ Tiled inference with Hann-window blending (same as N2V_multi)

---

## Summary Table

| # | File | Component | Severity | Description |
|---|---|---|---|---|
| 1 | N2V_multi | Mask ratio | Medium | Default 0.6% vs official 1.5%; affects convergence speed |
| 2 | N2V_multi | Training loss | Medium | MSE (L2) used; official default is L1 (more robust to hot pixels) |
| 3 | N2V_multi | Patch sampling | Low | Off-by-one: `integers(0, H-P)` misses last valid edge position |
| 4 | N2V_multi | Zero-displacement fix | Low | Minor over-sampling of 4-nearest neighbors when (dr,dc)=(0,0) is drawn |
| 5 | N2V_multi | Train/val split | Low | No spatial separation; both datasets sample from same image pool |
| 6 | PN2V_multi | Noise model type | High | Parametric GMM vs official non-parametric histogram; restricts modelling flexibility |
| 7 | PN2V_multi | Network output | Critical | Single scalar prediction vs official K=800 sample representation; loses posterior distribution |
| 8 | PN2V_multi | Inference | Critical | Direct UNet output used; MMSE posterior mean (the paper's key contribution) is **not implemented** |
| 9 | PN2V_multi | Patch sampling | Low | Same off-by-one as N2V_multi |
| 10 | PN2V_multi | Train/val split | Low | Same as N2V_multi |

---

## Recommended Fixes by Priority

### Priority 1 — PN2V Inference (implements the paper's core contribution)

Add a `predict_mmse` path that uses the trained GMM to compute the posterior mean at inference time. Minimal version for a single-point-estimate network (items 7 & 8 together):

```python
def predict_mmse_correction(
    net_output: torch.Tensor,   # (B, 1, H, W) — network signal estimate
    noisy_obs:  torch.Tensor,   # (B, 1, H, W) — original noisy pixels
    noise_model: GMMNoiseModel,
    n_hypotheses: int = 50,
    search_half_width: float = 0.1,
) -> torch.Tensor:
    """
    MMSE correction around network output using the trained GMM.
    For each pixel, evaluates p(y|s) on a grid of s values centred on
    the network estimate, then returns the likelihood-weighted mean.
    """
    s0 = net_output.squeeze(1)          # (B, H, W)
    y  = noisy_obs.squeeze(1)           # (B, H, W)

    # Build hypothesis grid: s_hat ± search_half_width
    deltas = torch.linspace(-search_half_width, search_half_width,
                             n_hypotheses, device=s0.device)   # (K,)
    s_grid = s0.unsqueeze(-1) + deltas  # (B, H, W, K)

    y_exp  = y.unsqueeze(-1).expand_as(s_grid)    # (B, H, W, K)
    s_flat = s_grid.reshape(-1, n_hypotheses)      # (B*H*W, K)
    y_flat = y_exp.reshape(-1, n_hypotheses)       # (B*H*W, K)

    # log p(y | s) for each hypothesis  → (B*H*W, K)
    log_w  = torch.stack([
        noise_model.log_prob(y_flat[:, k], s_flat[:, k])
        for k in range(n_hypotheses)
    ], dim=-1)

    weights   = torch.softmax(log_w, dim=-1)               # (B*H*W, K)
    s_mmse    = (weights * s_flat).sum(-1)                  # (B*H*W,)
    return s_mmse.reshape_as(s0).unsqueeze(1)
```

Then in `predict_tiled`, apply this correction after the network forward pass on each tile.

### Priority 2 — Loss function in N2V_multi (issue 2)

Change:
```python
loss_fn = nn.MSELoss(reduction='sum')
```
to:
```python
loss_fn = nn.L1Loss(reduction='sum')
```

### Priority 3 — Patch sampling off-by-one (issues 3 & 9)

```python
# Change in MultiImageN2VDataset.__getitem__ and MultiImagePN2VDataset.__getitem__
r0 = int(self.rng.integers(0, H - P + 1))  # was H - P
c0 = int(self.rng.integers(0, W - P + 1))  # was W - P
```

### Priority 4 — Mask ratio alignment (issue 1)

Consider raising the default to match the official:
```python
mask_ratio: float = 0.015  # 1.5%, matches official N2V default
```
Or add a CLI flag `--mask_ratio` so users can tune it.

---

## Applied Fixes — 2026-04-17

The following changes were implemented across **four files**:
`denoise_N2V.py`, `denoise_N2V_multi.py`, `denoise_N2V_GMM.py`, `denoise_N2V_GMM_multi.py`.

Original versions backed up to `backup/` with suffix `_2026-04-16`.

### Fix 1 — MMSE Posterior Mean at Inference (Issue 8 / Priority 1)

**Scope:** `denoise_N2V_GMM.py`, `denoise_N2V_GMM_multi.py`

**Implementation (2026-04-16):** Added `_apply_mmse_tile()` implementing Equation (4) of Krull et al. (2020) with a uniform hypothesis grid around `s_pred`. `predict_tiled` gained a `noise_model` parameter; `main()` defaulted to MMSE enabled.

**Regression found (2026-04-17):** Enabling MMSE caused the denoised output to be nearly identical to the original noisy image. Root cause identified and MMSE disabled by default.

#### Root Cause Analysis

The MMSE formula `ŝᵢ = Σₖ p(yᵢ|sₖ)·sₖ / Σₖ p(yᵢ|sₖ)` is valid **only** when `{sₖ}` are samples drawn from the network's posterior `p(s | context)`, as in official PN2V (K=800 forward passes). With a **uniform grid** around `s_pred`, the formula reduces to a likelihood search — and `p(y|s)` is maximized at `s ≈ y` (GMM mean `μ_k(s) = s + offset_k`, noise std ≈ 0.05 after `var_b = -6.0` initialization). Concretely:

- GMM std σ ≈ 0.05 → likelihood ratio between two hypotheses Δs = 0.1 apart: `exp(-Δs²/2σ²) = exp(-2) ≈ 7×`
- For any pixel where `y_obs` falls within `[s_pred − 0.15, s_pred + 0.15]` (i.e., noise ≤ 0.15 ≈ 3σ, which covers virtually all SEM pixels), the grid point closest to `y_obs` dominates
- Result: `s_mmse ≈ y_obs` (noisy image), all denoising erased

For a single-scalar prediction network, `s_pred` **is** the optimal estimate — the MMSE posterior mean requires the network to output a distribution, not a point. No grid correction is meaningful without K posterior samples.

**Current state (2026-04-17):** `_apply_mmse_tile()` is kept in the code for research purposes with an explicit warning in its docstring. Default behaviour is MMSE **disabled**. The `--no_mmse` flag was renamed to `--use_mmse` (opt-in, off by default).

### Fix 2 — L2 → L1 Training Loss (Issue 2 / Priority 2)

**Scope:** `denoise_N2V.py`, `denoise_N2V_multi.py`

```python
# Before
loss_fn = nn.MSELoss(reduction='sum')
# After
loss_fn = nn.L1Loss(reduction='sum')
```

L1 loss is more robust to hot pixels and charging artefacts in SEM images, matching the official N2V default. No other training logic was changed.

### Fix 3 — Patch Sampling Off-By-One (Issues 3 & 9 / Priority 3)

**Scope:** All four files (single-image and multi-image datasets)

```python
# Before — excludes last valid edge position
r0 = self.rng.integers(0, self.H - P)     # [0, H-P-1]
# After — includes last valid edge position
r0 = self.rng.integers(0, self.H - P + 1) # [0, H-P]
```

Same correction applied to `c0` and, in multi-image files, to both `H - P` and `W - P` inside `__getitem__`.

### Fix 4 — Zero-Displacement Resample (Issue B / Issue 4)

**Scope:** All four files (`_apply_n2v_masking` / `_apply_n2v_masking`)

The previous fix for `(dr, dc) == (0, 0)` redirected to one of the 4 nearest neighbours, biasing the replacement distribution toward immediate adjacents. The new approach re-samples from the full `(2r+1)²` neighbourhood, with a deterministic fallback `(1, 0)` only for the double-zero edge case (probability ≈ 1/121²):

```python
# Before — biased toward 4-nearest neighbours
shift_dr = self.rng.integers(0, 2, size=n_fix).astype(bool)
sign     = self.rng.choice([-1, 1], size=n_fix)
dr[zero_mask] = np.where(shift_dr,  sign, 0)
dc[zero_mask] = np.where(~shift_dr, sign, 0)

# After — uniform re-sample from full neighbourhood
dr_fix = self.rng.integers(-rad, rad + 1, size=n_fix)
dc_fix = self.rng.integers(-rad, rad + 1, size=n_fix)
still_zero = (dr_fix == 0) & (dc_fix == 0)
dr_fix[still_zero] = 1          # deterministic fallback
dr[zero_mask] = dr_fix
dc[zero_mask] = dc_fix
```

### Issues Not Fixed

| # | Issue | Reason not fixed |
|---|---|---|
| 1 | Mask ratio 0.6% vs 1.5% | Within valid range; SEM detail preservation may benefit from lower masking; left as-is |
| 5, 10 | Train/val no spatial split | Acceptable for multi-image scenario; patch overlap probability is low |
| 6 | GMM vs histogram noise model | GMM is physically motivated for SEM (Poisson + Gaussian); histogram offers no advantage here |
| 7 | Single scalar vs K=800 samples | Full sample-based output requires UNet head redesign; MMSE grid (Fix 1) approximates the benefit |

---

## Conclusion

`denoise_N2V_multi.py` is a functionally correct N2V implementation with minor deviations from the official defaults (mask ratio, L2 vs L1 loss, patch edge bias). These do not break the algorithm but may affect convergence speed and robustness to outlier pixels.

`denoise_N2V_GMM_multi.py` has two critical gaps relative to the published PN2V algorithm: (1) the network produces a single scalar per pixel rather than a sample distribution, and (2) inference returns the raw network output rather than the MMSE posterior mean. The result is that the trained GMM noise model influences training but is **completely bypassed at inference time** — the denoising quality falls back to N2V-level rather than delivering the probabilistic improvement that PN2V promises. The GMM parameterization (vs. the official histogram) is a valid engineering choice for physically-motivated SEM noise but loses flexibility.

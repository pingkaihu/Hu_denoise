# PN2V vs PPN2V: Technical Comparison

**Methods:** Probabilistic Noise2Void (PN2V) and Fully Unsupervised Probabilistic Noise2Void (PPN2V)  
**Authors:** Alexander Krull et al. (juglab, EMBL)  
**Papers:**
- PN2V — *Probabilistic Noise2Void: Unsupervised Content-Aware Denoising*, Frontiers in Computer Science 2020 ([arXiv:1906.00651](https://arxiv.org/abs/1906.00651))
- PPN2V — *Fully Unsupervised Probabilistic Noise2Void*, arXiv 2020 ([arXiv:1911.12291](https://arxiv.org/abs/1911.12291))

---

## 1. Background and Motivation

**Why PN2V exists (vs plain N2V):**  
Standard N2V trains a deterministic denoiser that outputs a single pixel value. It implicitly assumes a
symmetric, zero-mean noise distribution (it minimises MSE), which is a poor fit for Poisson or
Poisson-Gamma noise in fluorescence and electron microscopy. PN2V replaces the deterministic head with a
probabilistic one: the network outputs K=800 samples from the *prior* `p(s | context)`, and an explicit
noise model `p(y | s)` provides the likelihood. Training minimises the negative log-*evidence* (marginal
likelihood over all K samples), which forces the K outputs to collectively explain the observed pixel
value. The result is uncertainty-aware denoising: each pixel comes with a full posterior distribution.

**Why PPN2V exists (vs PN2V):**  
PN2V's noise model is a 2D histogram built from *external calibration images* — noisy snapshots of a
uniform scene with no sample present. In many microscopy workflows, acquiring such images is impractical
(phototoxic samples, no access to a blank slide, varying hardware conditions). PPN2V replaces the
non-parametric histogram with a *parametric GMM* and adds a **bootstrap mode** that estimates the GMM
from the image being denoised itself, using a preliminary N2V denoising as the pseudo-clean reference.
This makes PPN2V fully unsupervised: no calibration data of any kind is needed.

---

## 2. Similarities

Both methods share the same core probabilistic framework — the differences are confined to the noise
model representation and the calibration step.

| Shared aspect | Detail |
|---|---|
| **Network architecture** | U-Net (depth 3–4, base 32–64 channels); final `Conv1×1` head |
| **K=800 output channels** | Each channel = one sample `s_k` from the learned prior `p(s \| context)` |
| **Training loss** | Negative log-evidence: `L = −log((1/K) Σ_k p(y\|s_k))` |
| **Inference** | MMSE posterior mean: `Σ_k p(y\|s_k)·s_k / Σ_k p(y\|s_k)` |
| **Blind-spot masking** | N2V masking (~0.6% pixels/patch, random 5×5 neighbour replacement) |
| **Uncertainty output** | K samples = full approximate posterior; credible intervals computable |

The training loss and inference formula are identical. Only what goes into `p(y|s_k)` differs.

---

## 3. Key Differences

### 3.1 Noise Model: Histogram vs GMM

**PN2V — Non-parametric 2D histogram**

The noise model is a 256×256 conditional probability table. Row `i` represents `p(y | s ≈ i/255)`,
built by counting pixel pairs `(s_proxy, y_obs)` from calibration images and row-normalizing.

```python
# Building the histogram (PN2V / denoise_PN2V_juglab.py)
for r in range(H):
    for c in range(W):
        neighbors = [image[r-1,c], image[r+1,c], image[r,c-1], image[r,c+1]]
        s_proxy = np.mean(neighbors)          # 4-neighbor mean as low-noise proxy
        s_bin = int(s_proxy * (n_bins - 1))
        y_bin = int(image[r,c] * (n_bins - 1))
        hist[s_bin, y_bin] += 1
hist /= hist.sum(axis=1, keepdims=True)       # row-normalize → p(y|s)

# Likelihood query — bilinear interpolation (gradient flows through s)
s_float = s.clamp(0, 1) * (n_bins - 1)
s_lo = s_float.long().clamp(0, n_bins - 2)
frac  = s_float - s_lo.float()               # ← gradient here: d(frac)/d(s) = n_bins-1
p_lo  = hist_t[s_lo, y_bin]
p_hi  = hist_t[s_lo + 1, y_bin]
p     = p_lo * (1 - frac) + p_hi * frac
return torch.log(p.clamp(min=1e-30))
```

- **Pros:** Model-free — captures any noise shape, including bimodal or heavy-tailed distributions
- **Cons:** Requires ~100 calibration images for reliable statistics; degrades with sparse coverage at
  extreme intensity bins; fixed-resolution 256×256 grid

**PPN2V — Parametric signal-dependent GMM**

The noise model is a K-component Gaussian mixture where weights, means, and variances are all
polynomial functions of the signal `s`:

```
p(y | s) = Σ_{k=1}^{K_gmm} α_k(s) · N(y; μ_k(s), σ²_k(s))

where:
  α_k(s) = softmax( Σ_j a_kj · s^j )       # signal-dependent weights
  μ_k(s) = Σ_j b_kj · s^j                  # signal-dependent means  (often μ_k ≈ s)
  σ²_k(s) = exp( Σ_j c_kj · s^j )          # log-linear variance (captures Poisson scaling)
```

Simplified implementation (as in `denoise_PN2V.py` / `GMMNoiseModel`):

```python
# Signal-dependent GMM — simplified version (degree-1 polynomial)
log_w     = F.log_softmax(self.log_weights, dim=0)   # (K_gmm,)
mu        = s.unsqueeze(-1) + self.mean_offsets       # (N, K_gmm)  ← μ_k(s) = s + offset_k
log_var   = self.var_a * s.unsqueeze(-1) + self.var_b # (N, K_gmm)  ← log σ²_k(s) = a_k·s + b_k
var       = log_var.exp() + 1e-8                      # (N, K_gmm)
log_gauss = -0.5 * ((y.unsqueeze(-1) - mu)**2 / var
            + log_var + math.log(2 * math.pi))        # (N, K_gmm)
log_prob  = (log_w + log_gauss).logsumexp(dim=-1)     # (N,)
```

- **Pros:** Only ~30 float parameters; extrapolates to unseen intensity ranges; robust with few calibration
  samples; closed-form gradients
- **Cons:** May mis-fit strongly non-Gaussian noise (e.g., salt-and-pepper); model capacity limited by
  polynomial degree and K_gmm

---

### 3.2 Calibration: Required vs Bootstrap

**PN2V** — external calibration is mandatory. You must provide a set of images of a uniform scene (no
sample) captured under identical conditions. The histogram is built from these and frozen before training.

```
Calibration images → build_histogram(images) → hist[256,256] (fixed) → train network
```

**PPN2V** — three operating modes:

| Mode | Calibration source | Noise model |
|---|---|---|
| **GMM (calibration)** | External calibration images | GMM fit to `(calibration pairs)` |
| **Boot GMM** | None — N2V output used as pseudo-GT | GMM fit to `(N2V(y), y)` pairs |
| **Boot Hist** | None — N2V output used as pseudo-GT | Histogram from `(N2V(y), y)` pairs |

Bootstrap procedure (Boot GMM):

```python
# Step 1: train a standard N2V on the noisy image (fast, ~50 epochs)
n2v_model = train_n2v(noisy_image, epochs=50)
pseudo_clean = n2v_model.predict(noisy_image)         # shape (H, W)

# Step 2: fit GMM to (pseudo_clean, noisy) pixel pairs
s_proxy = pseudo_clean.flatten()
y_obs   = noisy_image.flatten()
gmm_params = fit_gmm(s_proxy, y_obs)                  # α_k, μ_k, σ²_k

# Step 3: train PN2V network with the fitted GMM fixed
train_pn2v(noisy_image, noise_model=gmm_params, epochs=200)
```

The N2V pseudo-clean is imperfect (it is a biased estimator under non-Gaussian noise), but in practice
the GMM fit is robust enough that Boot GMM achieves results very close to calibrated PN2V.

---

### 3.3 Training Loss (identical formula, different `p(y|s)`)

Both methods minimise:

```
L = −(1/N) Σ_{masked i} log( (1/K) Σ_{k=1}^{K} p(y_i | s_i^k) )
  = −(1/N) Σ_i [ logsumexp_k( log p(y_i | s_i^k) ) − log K ]
```

In PyTorch:

```python
# pred: (B, K, H, W) — K output channels from UNet
s_samp = pred.permute(0,2,3,1)[mask_2d]           # (N_masked, K)
y_exp  = y_obs.unsqueeze(1).expand(N_masked, K)   # (N_masked, K)

# log_likelihood dispatches to histogram (PN2V) or GMM (PPN2V)
log_liks = noise_model.log_likelihood(
    y_exp.reshape(-1), s_samp.reshape(-1)
).reshape(N_masked, K)                            # (N_masked, K)

log_ev = torch.logsumexp(log_liks, dim=1) - math.log(K)
loss   = -log_ev.mean()
```

The only difference is `noise_model.log_likelihood` — histogram bilinear lookup for PN2V, GMM
closed-form for PPN2V.

---

### 3.4 Inference: MMSE Posterior Mean (identical)

Both use the same importance-weighted estimate:

```python
# samples: (K, N) — K UNet predictions for each of N pixels
# y_flat:  (N,)   — observed noisy pixel values

log_liks = noise_model.log_likelihood(
    y_flat.unsqueeze(0).expand(K, N).reshape(-1),
    samples.reshape(-1),
).reshape(K, N)

weights  = torch.softmax(log_liks, dim=0)       # (K, N) — importance weights
mmse     = (weights * samples).sum(dim=0)        # (N,)   — posterior mean
```

The posterior mean is the optimal MMSE estimator under squared loss. Both methods also output a
"prior mean" = `samples.mean(dim=0)` as a diagnostic (ignores noise model weighting).

---

### 3.5 Summary Comparison Table

| Aspect | PN2V | PPN2V |
|---|---|---|
| Noise model | Non-parametric 2D histogram (256×256) | Parametric GMM (signal-dependent) |
| Calibration | **Required** — external images of blank scene | Optional — Boot modes use N2V pseudo-GT |
| Fully unsupervised | No | **Yes** (Boot GMM / Boot Hist) |
| Histogram building | 4-neighbor mean proxy over calibration set | Not used (Boot) / same for Boot Hist |
| Likelihood computation | Bilinear table lookup — `O(1)` | Σ of K_gmm Gaussian PDFs — `O(K_gmm)` |
| Memory (noise model) | 256 KB fixed array | ~30 floats (~0.1 KB) |
| Robustness to limited calibration | Low — sparse bins → noisy histogram | High — GMM extrapolates |
| Model expressivity | Very high — no distributional assumption | Moderate — limited by polynomial degree |
| Bootstrap mode | Not available | Boot GMM and Boot Hist |
| Performance vs supervised | Close to CARE | Very close to PN2V (Boot GMM) |

---

## 4. When to Choose Which

**Use PN2V when:**
- You have 50–100 calibration images from your imaging system (easy to acquire for most microscopes)
- The noise distribution is strongly non-Gaussian or bimodal (histogram captures it; GMM may not)
- You want the most faithful representation of your camera's noise without distributional assumptions
- Implementation is already available (`denoise_PN2V_juglab.py` in this project)

**Use PPN2V when:**
- Calibration images are impractical to acquire (photosensitive samples, inaccessible microscope setup)
- You want a single-image, zero-external-data workflow (`Boot GMM` mode)
- Hardware or acquisition conditions vary between sessions (GMM is more robust to distribution shift)
- You need a compact, portable noise model (30 parameters vs 256×256 array)

**Performance note (from the PPN2V paper):** Boot GMM achieves "very similar quality" to calibrated PN2V
on all tested datasets. The quality gap between calibrated GMM (PPN2V) and histogram (PN2V) is
dataset-dependent — histogram wins on some datasets, GMM on others.

---

## 5. Detailed Analysis of `denoise_PN2V.py` vs Official PPN2V

File: [denoise_PN2V.py](../denoise_PN2V.py)

**Overall verdict:** `denoise_PN2V.py` is closest to **PPN2V Boot GMM** (same noise model family, same
self-calibration philosophy), but is missing the three features that make PPN2V probabilistic. It has
six concrete divergences from the official implementation. Some are deliberate simplifications with
practical benefits; others are genuine capability losses.

---

### Divergence 1 — Network Output: 1 Channel vs K=800 Channels

| | Official PPN2V | `denoise_PN2V.py` |
|---|---|---|
| UNet head | `Conv2d(f, 800, kernel_size=1)` → `(B, 800, H, W)` | `Conv2d(f, 1, kernel_size=1)` → `(B, 1, H, W)` |
| Interpretation | 800 samples from the prior `p(s \| context)` | Single point estimate of `s` |

**Official PPN2V:**
```python
# Final head produces K=800 samples per pixel
self.head = nn.Conv2d(base_features, K=800, kernel_size=1)
pred = model(noisy_in)     # (B, 800, H, W)
# Each channel k = one hypothesis for the true signal
```

**`denoise_PN2V.py`:**
```python
# Final head produces 1 scalar per pixel (line 134)
self.head = nn.Conv2d(f, in_channels, kernel_size=1)
pred = model(noisy_in)     # (B, 1, H, W)
```

**Impact — Bad:**
- The K=800 outputs are the mechanism by which PPN2V represents uncertainty. The network learns a
  *diversity of explanations* for each pixel: some samples represent high signal (bright), others low
  signal (dark), and the noise model weights them by how consistent each is with the observed noisy value.
  With K=1, this diversity is impossible — the network can only commit to a single prediction.
- Concretely: under Poisson noise, a bright pixel could be caused by a genuinely bright signal **or** by
  a noise spike on a moderate signal. K=800 sampling allows the posterior to reflect both; K=1 forces a
  hard choice.
- No uncertainty map can be produced (loss of diagnostic capability).

**Impact — Good (practical trade-off):**
- Memory: K=800 increases GPU memory by 800× for the output tensor. At batch=32, patch=64:
  `32 × 800 × 64 × 64 × 4 bytes ≈ 419 MB` just for the output, vs 0.5 MB for K=1. K=1 makes the
  script feasible on any modern GPU without memory tuning.
- Speed: 800× fewer output channels → significantly faster forward pass and gradient computation.
- For denoising quality alone (not uncertainty), a well-trained K=1 model still produces good
  point-estimate denoising; it just cannot provide posteriors.

---

### Divergence 2 — Training Loss: Direct NLL vs Log-Evidence

| | Official PPN2V | `denoise_PN2V.py` |
|---|---|---|
| Loss | `−log((1/K) Σ_k p(y\|s_k))` = log-evidence | `−log p_GMM(y\|s_pred)` = direct NLL |

**Official PPN2V:**
```python
# Negative log-marginal-likelihood over K samples (log-evidence)
log_liks = noise_model.log_prob(y_exp, s_samp)   # (N_masked, K)
log_ev   = torch.logsumexp(log_liks, dim=1) - math.log(K)
loss     = -log_ev.mean()
```

**`denoise_PN2V.py` (line 525):**
```python
# Direct negative log-likelihood of single prediction
loss = noise_model.nll_loss(y_obs, s_pred)        # = -log p_GMM(y|s_pred).mean()
```

**Impact — Neutral (mathematically equivalent given K=1):**

With K=1, the log-evidence loss reduces exactly to the direct NLL:

```
−log((1/1) Σ_{k=1}^{1} p(y|s_k)) = −log p(y|s_1)
```

So the loss function is **not a divergence** from the official method — it is the mathematically
correct loss for a K=1 network. The change looks different in code but is equivalent.

The deeper issue is that the *loss landscape* differs when K=1 vs K=800. With K=800, the logsumexp
is a smooth upper bound that encourages the network to spread probability mass across diverse
hypotheses (implicit diversity regularisation). With K=1, the loss is a strict point-wise penalty
that drives the network toward a single maximum-likelihood estimate. Both are correct; they encode
different inductive biases.

---

### Divergence 3 — Mixture Weights: Constant vs Signal-Dependent

| | Official PPN2V | `denoise_PN2V.py` |
|---|---|---|
| α_k(s) | `softmax(polynomial(s))` — weights vary with signal | `softmax(log_weights)` — **constant** for all s |

**Official PPN2V:**
```python
# Weights are polynomial functions of signal s (higher-order PPN2V)
alpha = softmax(W_alpha @ poly_features(s), dim=-1)  # (N, K_gmm)
```

**`denoise_PN2V.py` (lines 236–237):**
```python
log_w = F.log_softmax(self.log_weights, dim=0)  # (K,) — same weights for all s values
mu    = s + self.mean_offsets                    # (N, K) — means DO depend on s
```

**Impact — Mild Bad:**
- Signal-dependent weights would allow, e.g., "at high signal levels, shot noise dominates (component 1
  gets weight 0.9); at near-zero signal, read noise dominates (component 2 gets weight 0.7)." This is the
  physically correct behaviour for Poisson + Gaussian noise mixtures.
- Constant weights mean the mixture always blends the same proportions of shot and read noise regardless
  of the actual signal level. This is a simplification but not catastrophic — the log-linear variance
  `σ²_k(s) = exp(a_k·s + b_k)` already handles the signal-dependent variance correctly, and in practice
  the constant-weight GMM fits SEM noise well enough.
- For truly multi-regime noise (very dark background + very bright foreground with different noise
  character), constant weights may produce a compromise model that is suboptimal in both regimes.

**Impact — Good (practical):**
- Fewer parameters: 3 scalars vs 3×(polynomial degree+1) scalars.
- No gradient instability from the weight polynomial at extreme signal values.

---

### Divergence 4 — GMM Training: Joint Fine-Tuning vs Frozen After Calibration

| | Official PPN2V | `denoise_PN2V.py` |
|---|---|---|
| GMM during network training | **Frozen** — fitted once, then fixed | **Co-trained** at 0.1× lr (`gmm_lr_scale=0.1`) |

**Official PPN2V Boot GMM:**
```python
# Stage 1: fit GMM from (N2V_output, noisy) pairs
gmm.fit(n2v_output.flatten(), noisy.flatten())

# Stage 2: train PN2V network — GMM parameters are FROZEN
for epoch in range(n_epochs):
    pred = network(noisy_masked)           # K=800 outputs
    log_lik = gmm.log_prob(y, pred)        # GMM fixed
    loss = -logsumexp(log_lik, dim=1).mean()
    # Only network.parameters() receive gradients
```

**`denoise_PN2V.py` (lines 489–493, 505):**
```python
# Both UNet AND GMM parameters are optimised simultaneously
optimizer = optim.Adam([
    {'params': model.parameters(),       'lr': learning_rate},
    {'params': noise_model.parameters(), 'lr': learning_rate * gmm_lr_scale},  # 0.1×
])
# Both receive gradients every iteration
```

**Impact — Bad (risk of co-adaptation):**
- When the GMM is free to move during training, a pathological equilibrium is possible: the GMM learns
  to assign high likelihood to whatever the network predicts, while the network learns to predict
  whatever minimises the current GMM's NLL. This circular dynamic can cause the pair to drift to a
  trivial solution (e.g., network outputs a constant near 0.5; GMM places all mass there).
- In the worst case, the co-trained system produces a denoised output with less structure than the
  raw input because the GMM accommodates the network's errors rather than constraining them.

**Impact — Good (adaptive refinement):**
- The 4-neighbor proxy used during GMM pre-training underestimates noise variance (averaging reduces
  variance by a factor of ~2–4). Starting joint training allows the GMM to correct this bias once
  the network starts providing better signal estimates.
- In practice, the `gmm_lr_scale=0.1` effectively regularises the GMM refinement to be slow and
  small relative to network updates, limiting the co-adaptation risk. The 300-epoch pre-training
  provides a strong starting point that constrains the GMM's drift.

---

### Divergence 5 — GMM Calibration Source: 4-Neighbor Mean vs N2V Pseudo-Clean

| | Official PPN2V Boot GMM | `denoise_PN2V.py` |
|---|---|---|
| Signal proxy for GMM fitting | N2V output (denoised image, run first) | 4-neighbor mean of noisy image |
| Training cost | 2 training phases (N2V + PN2V) | 1 training phase (GMM pretrain + joint) |

**Official PPN2V Boot GMM:**
```python
# Phase 1: full N2V training (~50 epochs) to get a pseudo-clean image
n2v_model = train_n2v(noisy_image, epochs=50)
pseudo_clean = n2v_model.predict(noisy_image)   # denoised — much better signal proxy

# Phase 2: fit GMM to (pseudo_clean, noisy) pairs
gmm.fit_mle(s=pseudo_clean.flatten(), y=noisy.flatten())
```

**`denoise_PN2V.py` (lines 400–415):**
```python
# 4-neighbor mean as signal proxy — no separate N2V run
kernel = torch.zeros(1, 1, 3, 3)
kernel[0,0,0,1] = kernel[0,0,1,0] = kernel[0,0,1,2] = kernel[0,0,2,1] = 0.25
s_proxy = F.conv2d(img_t, kernel, padding=1).squeeze()   # (H, W)
# Fit GMM to (s_proxy, y_obs) pairs
```

**Impact — Moderate Bad (noisier initialization):**
- The 4-neighbor mean has variance ≈ σ²/4 (averages 4 independent noisy pixels), so it is a noisier
  signal proxy than N2V output which suppresses noise across the full receptive field (~64×64 pixels).
- This means the fitted GMM sees a signal proxy that still contains substantial noise, causing it to
  underestimate the true noise amplitude. The histogram `p(y|s_proxy)` is widened by the residual
  noise in `s_proxy`.
- This manifests as slightly over-smoothed GMM pre-training: the model thinks noise is larger than it
  is, leading to a more aggressive NLL landscape early in training.

**Impact — Significant Good (simplicity):**
- Eliminates the entire N2V pre-training phase (saves ~50 epochs × full training time).
- No dependency on N2V hyperparameter choices (patch size, masking ratio) affecting the GMM quality.
- The GMM is a rough initialisation anyway; the 300-epoch pre-training and subsequent joint fine-tuning
  largely overcome the proxy's imperfection in practice.

---

### Divergence 6 — Inference: Raw Scalar vs MMSE Posterior Mean

| | Official PPN2V | `denoise_PN2V.py` |
|---|---|---|
| Inference output | MMSE: `Σ_k p(y\|s_k)·s_k / Σ_k p(y\|s_k)` | Raw UNet scalar (default) |
| MMSE available | Yes — 800 posterior samples feed directly | `--use_mmse` flag → **broken** (see below) |

**Official PPN2V:**
```python
pred    = model(noisy)                     # (B, K=800, H, W)
log_w   = gmm.log_prob(y_tiled, pred)     # (B, K, H, W)
weights = torch.softmax(log_w, dim=1)      # importance weights
mmse    = (weights * pred).sum(dim=1)      # (B, H, W) — posterior mean
```

**`denoise_PN2V.py` (line 693):**
```python
preds = model(batch_t).squeeze(1).cpu().numpy()   # (B, H, W) — raw scalar, no weighting
```

The script includes `_apply_mmse_tile()` (lines 576–636) with a **uniform hypothesis grid**:

```python
# Attempt to replicate MMSE using a grid around the single prediction
deltas  = torch.linspace(-0.15, 0.15, n_hypotheses=50)   # fixed grid ±0.15
s_grid  = s_pred.unsqueeze(1) + deltas                    # (N, 50) — NOT posterior samples
log_w   = gmm.log_prob(y_exp, s_grid)                     # (N, 50)
weights = torch.softmax(log_w, dim=1)
s_mmse  = (weights * s_grid).sum(dim=1)                   # (N,) — weighted grid average
```

**Why this is broken (the docstring explains it):**

The MMSE formula requires that the `{s_k}` are **samples from the learned prior `p(s|context)`**.
In that case, the weighting by `p(y|s_k)` implements Bayesian posterior inference.

A uniform grid is not drawn from `p(s|context)` — it is a deterministic set of points centered on
the prediction. The GMM likelihood `p(y|s)` peaks at `s ≈ y_obs` (the noisy value), so with a
tight-enough noise model, the grid weighting simply selects the grid point closest to the noisy
observation — producing `s_mmse ≈ y_obs` (the original noisy image with no denoising).

**Impact — Significant Bad:**
- No proper MMSE inference means the script cannot produce the posterior-mean output that is the core
  scientific contribution of PN2V/PPN2V. The raw scalar output is just a point estimate — identical in
  character to what plain N2V produces, but with a GMM-regularised loss instead of MSE.
- The `--use_mmse` flag is available but actively harmful: enabling it makes the output *worse* than
  the raw prediction by pulling it toward the noisy input.

**Impact — Neutral (for point-estimate denoising):**
- If the goal is simply to get a clean-looking image (not uncertainty quantification), the raw UNet
  prediction is already a reasonable denoising. The GMM loss guides the network to produce outputs
  that are consistent with the noise model, which achieves the same goal via a different path.

---

### Overall Impact Summary

| Divergence | Direction | Severity | Impact |
|---|---|---|---|
| K=1 output (vs K=800) | Simplification | **High** | Loses full posterior; breaks MMSE; saves ~800× memory |
| Constant mixture weights | Simplification | Low–Medium | Slightly less expressive noise model; no instability |
| Joint GMM training | Deviation | Medium | Risk of co-adaptation; allows adaptive refinement |
| 4-neighbor proxy (vs N2V) | Simplification | Low–Medium | Noisier GMM init; saves one full training phase |
| Broken MMSE inference | Bug (given K=1) | Medium | `--use_mmse` worsens output; raw scalar is fine |
| No signal-dependent weights | Simplification | Low | Minor expressivity loss; simpler, more stable |

**Summary sentence:** `denoise_PN2V.py` trades the full probabilistic machinery of PPN2V for a much
simpler, memory-efficient, single-pass implementation that produces a deterministic point estimate
rather than a posterior distribution. For pure denoising quality the trade-off is reasonable; for
uncertainty quantification or rigorous Bayesian inference it is inadequate.

---

### Would a PPN2V-faithful script be straightforward?

Yes — `denoise_PN2V.py`'s `GMMNoiseModel` is reusable. The minimum required changes:

```python
# 1. Change UNet head: 1 → K channels
self.head = nn.Conv2d(f, K, kernel_size=1)           # K=800

# 2. Change training loss: NLL → log-evidence
log_liks = noise_model.log_prob(y_exp, s_samp)       # (N_masked, K)
loss = -(torch.logsumexp(log_liks, dim=1) - math.log(K)).mean()

# 3. Change inference: raw → MMSE
pred    = model(noisy_t)                              # (B, K, H, W)
log_w   = noise_model.log_prob(y_tiled, pred)         # (B, K, H, W)
weights = torch.softmax(log_w, dim=1)
mmse    = (weights * pred).sum(dim=1)                 # (B, H, W)

# 4. (Optional) bootstrap: run N2V first → fit GMM → freeze GMM → train network
```

The first three changes are ~15 lines of code. The main obstacle is GPU memory: K=800 at batch=32,
patch=64 requires ~420 MB for the output tensor alone, making it impractical on <8 GB GPUs without
reducing batch size to ≤4 or using mixed precision.

---

## 6. References

| Paper | arXiv | Local PDF |
|---|---|---|
| PN2V (Krull et al., Frontiers 2020) | [1906.00651](https://arxiv.org/abs/1906.00651) | [pn2v_krull2020.pdf](../reference/pn2v_krull2020.pdf) |
| PPN2V (Krull et al., 2020) | [1911.12291](https://arxiv.org/abs/1911.12291) | [ppn2v_krull2020.pdf](../reference/ppn2v_krull2020.pdf) |
| N2V (Krull et al., CVPR 2019) | [1811.10980](https://arxiv.org/abs/1811.10980) | [noise2void_krull2019.pdf](../reference/noise2void_krull2019.pdf) |

**juglab GitHub repos:**
- PN2V: https://github.com/juglab/pn2v
- PPN2V: https://github.com/juglab/PPN2V

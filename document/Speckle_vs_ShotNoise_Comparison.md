# Speckle vs. Shot Noise 去噪策略比較
### 基於 N2N 家族的理論分析與 SEM 實作指南

---

## 目錄

1. [噪聲物理本質比較](#1-噪聲物理本質比較)
2. [統計模型深度分析](#2-統計模型深度分析)
3. [N2N 家族相容性對照](#3-n2n-家族相容性對照)
4. [橋接策略比較](#4-橋接策略比較)
5. [SEM 場景推薦方法](#5-sem-場景推薦方法)
6. [完整程式碼](#6-完整程式碼)
7. [方法選擇決策樹](#7-方法選擇決策樹)
8. [效能評估指標](#8-效能評估指標)
9. [綜合比較總表](#9-綜合比較總表)
10. [GitHub 資源](#10-github-資源)

---

## 1. 噪聲物理本質比較

### Shot Noise（散粒噪聲）

來源於光子或電子**計數的量子統計波動**。無論偵測器設計多完美，這是物理上無法消除的基本極限——每個電子是否被偵測到，本質上是獨立的隨機事件。

在 SEM 中，低電子束劑量（低 keV、高掃描速度、輻射敏感樣品）時，每個像素偵測到的電子數很少，Poisson 統計效應顯著。

**關鍵性質：**
- 像素間**統計獨立**：一個像素的計數值不影響鄰近像素
- 方差等於均值：`Var[y] = E[y] = λ`（光子數越少，相對噪聲越大）
- 高計數時趨近高斯分佈（中心極限定理）

### Speckle Noise（散斑噪聲）

來源於相干成像系統中電磁波的**干涉效應**。同一解析單元內多個散射體的波相位隨機疊加，形成明暗交替的 grain pattern。

在 SEM 中，低加速電壓、特定材料表面（粗糙、多晶）下的二次電子訊號中可見，本質上是**相干性引起的訊號本身波動**，而非計數統計。

**關鍵性質：**
- 像素間**空間相關**：相鄰像素的噪聲值不獨立，形成 grain
- 乘性模型：強訊號區的 speckle 也強
- Gamma / Rayleigh 分佈，非高斯

### 直覺對比

```
Shot noise：  每個像素各自獨立地「骰子一擲」，結果完全不看鄰居
Speckle：     像一塊布料的紋理，局部區域的明暗是連動的
```

---

## 2. 統計模型深度分析

### Shot Noise 的 Poisson 模型

```
y ~ Poisson(x)

性質：
  E[y | x]   = x          ← 期望等於真實訊號
  Var[y | x] = x          ← 方差等於均值（訊號相依）
  空間相關性：像素間獨立   ← blind-spot 假設完整成立

高計數極限（λ >> 1）：
  Poisson(λ) → N(λ, λ)   ← 趨近高斯，MSE loss 接近最優
```

**對 N2N 家族的影響：**
- 期望等價性 `E[y|x] = x` 直接成立 → N2N 理論保證完整
- 像素空間獨立 → N2V blind-spot 假設完整成立
- 唯一問題：Poisson ≠ 高斯，MSE loss 不是嚴格最優估計子

### Speckle 的 Gamma 乘性模型

```
y = x · n，  n ~ Gamma(L, 1/L)

性質：
  E[y | x]   = x          ← 期望等於真實訊號（與 shot 相同）
  Var[y | x] = x² / L     ← 方差與訊號平方成正比
  空間相關性：鄰近像素相關  ← blind-spot 假設受挑戰
  L = ENL（等效視數）：L 越大，speckle 越弱

```

**對 N2N 家族的影響：**
- 期望等價性成立 → N2N 框架在原理上可延伸
- 但乘性結構 + 空間相關 → N2V、R2R 假設破裂
- 需要額外的橋接策略（log 轉換或 GR2R）

### 兩種噪聲的假設違反程度

| N2N 假設 | Shot Noise | Speckle |
|---|---|---|
| `E[y\|x] = x` | ✅ 完全成立 | ✅ 完全成立 |
| 像素空間獨立 | ✅ 完全成立 | ❌ grain pattern 相關 |
| 加性噪聲 | ⚠️ 近似（高計數）| ❌ 乘性 |
| 高斯分佈 | ⚠️ 近似（高計數）| ❌ Gamma 分佈 |
| MSE 最優估計 | ⚠️ 近似 | ❌ 不適用 |

---

## 3. N2N 家族相容性對照

### Noise2Noise（N2N 原版）

| | Shot Noise | Speckle |
|---|---|---|
| 理論相容性 | ✅ `E[y\|x] = x` 成立 | ✅ 同上，期望等價 |
| 實務限制 | 需要同場景兩次觀測 | 同左，且更難確保獨立 |
| SEM 可行性 | 低劑量下可重複掃描 | 相干性使兩次觀測相關 |

### Noise2Void / N2V2

| | Shot Noise | Speckle |
|---|---|---|
| 像素獨立假設 | ✅ **完整成立** | ❌ grain 使假設破裂 |
| 直接使用 | ✅ **可直接使用** | ❌ 效果不佳 |
| 修改後使用 | — | ⚠️ Structured N2V 可部分改善 |
| 主要風險 | 低計數時 MSE 偏差 | 空間相關噪聲殘留 |

### Probabilistic N2V（PN2V）

| | Shot Noise | Speckle |
|---|---|---|
| 噪聲模型 | ✅ 直接支援 Poisson | ✅ 支援任意分佈（含 Gamma）|
| 資料需求 | 單張影像即可 | 單張影像即可 |
| 空間相關問題 | 無此問題 | 仍有 grain 殘留風險 |
| 推薦程度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

### R2R / GR2R

| | Shot Noise | Speckle |
|---|---|---|
| 原版 R2R | ❌ 僅限高斯 | ❌ 僅限高斯 |
| GR2R | ✅ 延伸至 Poisson | ✅ 延伸至 Gamma |
| 需求 | 需知道 Poisson 參數（λ）| 需知道 ENL 值（L）|

### Self2Self

| | Shot Noise | Speckle |
|---|---|---|
| 假設 | 無分佈假設 | 無分佈假設 |
| 空間相關 | 像素獨立，Dropout 有效 | ⚠️ 強相關時效果不穩定 |
| 推薦程度 | ⭐⭐⭐（N2V 更快）| ⭐⭐⭐⭐⭐（首選）|

---

## 4. 橋接策略比較

### Shot Noise 的橋接：Anscombe Transform（VST）

Shot noise（Poisson）有精確的**方差穩定化轉換（Variance-Stabilizing Transform）**：

```
Anscombe 轉換：
  A(y) = 2 · √(y + 3/8)

性質：
  A(y) ≈ N(2√x, 1)        ← 方差穩定為 1，與 x 無關
  適用條件：λ > 20（中高計數）

精確無偏反轉換（Makitalo & Foi, 2013）：
  A⁻¹(z) = (z/2)² - 3/8
```

轉換後的 shot noise ≈ **單位方差高斯加性噪聲**，所有為高斯設計的 N2N 方法全部適用，且無任何理論損失。

**與 Speckle 的 Log 轉換對比：**

| 性質 | Shot Noise：Anscombe | Speckle：Log 轉換 |
|---|---|---|
| 轉換目標 | Poisson → 近似高斯 | 乘性 → 加性 |
| 轉換精確度 | 高（有最優反轉換公式）| 近似（log-Gamma 非嚴格高斯）|
| 低計數時 | ⚠️ λ < 20 時失準 | △ 仍有效 |
| 廣義版本 | 廣義 Anscombe（GAT）處理混合噪聲 | 對數域 + Structured N2V |

### 兩種轉換的程式碼

```python
import numpy as np

# === Shot Noise：Anscombe VST ===
def anscombe(x):
    """Poisson 噪聲方差穩定化，轉換後 ≈ N(2√λ, 1)"""
    return 2.0 * np.sqrt(np.maximum(x, 0) + 3.0 / 8.0)

def inverse_anscombe(z):
    """精確無偏反轉換（推薦使用此版本，優於直接平方）"""
    return (z / 2.0) ** 2 - 3.0 / 8.0

def generalized_anscombe(x, mu, sigma, gain=1.0):
    """廣義 Anscombe：處理 Poisson-Gaussian 混合噪聲
    mu: 高斯讀出噪聲均值, sigma: 讀出噪聲標準差, gain: 偵測器增益
    """
    return (2.0 / gain) * np.sqrt(
        np.maximum(gain * x + gain**2 * (3.0/8.0) + sigma**2 - gain * mu, 0)
    )

# === Speckle：Log 轉換 ===
def log_transform(x):
    """乘性 → 加性：log(y) = log(x) + log(n)"""
    return np.log1p(x)   # log(1 + x)，數值穩定

def inverse_log_transform(z):
    return np.expm1(z)   # exp(z) - 1


# === 使用範例 ===
image = ...   # 你的 SEM 影像，float32，已正規化

# Shot noise 流程
img_vst = anscombe(image)
# → 套用任意高斯去噪器訓練 / 推論
# denoised_vst = your_denoiser(img_vst)
# denoised = inverse_anscombe(denoised_vst)

# Speckle 流程
img_log = log_transform(image)
# → 套用任意去噪器訓練 / 推論
# denoised_log = your_denoiser(img_log)
# denoised = inverse_log_transform(denoised_log)
```

---

## 5. SEM 場景推薦方法

### Shot Noise 推薦排序

**場景條件：** 低電子劑量，像素計數值偏低，噪聲看起來是隨機顆粒感。

#### 方案 A：PN2V（最推薦）

直接對 Poisson 分佈建模，不需要任何前處理轉換，理論最嚴謹。

```python
from careamics import CAREamist
from careamics.config import create_n2v_configuration
from careamics.models.noise_models import GaussianMixtureNoiseModel
import numpy as np
import tifffile

# 載入影像
image = tifffile.imread("sem_shot.tif").astype(np.float32)
train_data = image[np.newaxis, ...]   # (1, H, W)

# 從影像本身估計 Poisson noise model（無需額外校準資料）
noise_model = GaussianMixtureNoiseModel.create_from_images(
    images=train_data,
    num_gaussians=3,
)

config = create_n2v_configuration(
    experiment_name="sem_shot_pn2v",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=128,
    num_epochs=100,
)

careamist = CAREamist(source=config, noise_model=noise_model)
careamist.train(train_source=train_data, val_percentage=0.1)

denoised = careamist.predict(
    source=train_data, data_type="array", axes="YX",
    tile_size=[256, 256], tile_overlap=[48, 48],
)
denoised = np.squeeze(denoised)
tifffile.imwrite("denoised_pn2v.tif", denoised)
```

#### 方案 B：Anscombe + N2V（最快上手）

三行改動，利用現有 N2V 環境：

```python
import numpy as np
import tifffile
from careamics import CAREamist
from careamics.config import create_n2v_configuration

image = tifffile.imread("sem_shot.tif").astype(np.float32)

# === 前處理：Anscombe VST ===
img_vst = 2.0 * np.sqrt(np.maximum(image, 0) + 3.0 / 8.0)
img_norm = (img_vst - img_vst.min()) / (img_vst.max() - img_vst.min())
# ===========================

config = create_n2v_configuration(
    experiment_name="sem_shot_anscombe_n2v",
    data_type="array", axes="YX",
    patch_size=[64, 64], batch_size=128, num_epochs=100,
)
careamist = CAREamist(source=config)
careamist.train(train_source=img_norm[np.newaxis], val_percentage=0.1)

denoised_vst = np.squeeze(careamist.predict(
    source=img_norm[np.newaxis], data_type="array", axes="YX",
    tile_size=[256, 256], tile_overlap=[48, 48],
))

# === 後處理：精確無偏反轉換 ===
denoised_vst_rescaled = denoised_vst * (img_vst.max() - img_vst.min()) + img_vst.min()
denoised = (denoised_vst_rescaled / 2.0) ** 2 - 3.0 / 8.0
# ==============================

tifffile.imwrite("denoised_anscombe_n2v.tif", denoised.astype(np.float32))
```

#### 方案 C：N2V 直接使用（最省事）

Shot noise 的像素獨立性使 N2V 可以直接套用，不需任何修改。效果略遜於 PN2V，但訓練速度最快：

```python
# 與一般 N2V 流程完全相同，無需任何修改
# 僅需確保正規化到 [0, 1]
image_norm = (image - image.min()) / (image.max() - image.min())
# 接續標準 N2V 訓練流程...
```

#### 方案 D：VST + BM3D（無需訓練基準）

```python
import bm3d
import numpy as np

def vst_bm3d(image, sigma=1.0):
    """Anscombe + BM3D + 精確反轉換"""
    img_vst = 2.0 * np.sqrt(np.maximum(image, 0) + 3.0 / 8.0)
    denoised_vst = bm3d.bm3d(img_vst, sigma_psd=sigma,
                              stage_arg=bm3d.BM3DStages.ALL_STAGES)
    return (denoised_vst / 2.0) ** 2 - 3.0 / 8.0

result = vst_bm3d(image, sigma=1.0)
```

---

### Speckle 推薦排序

**場景條件：** 相干成像特性，影像出現 grain pattern，強訊號區噪聲也強。

#### 方案 A：Self2Self（首選）

```python
# git clone https://github.com/scut-mingqinchen/Self2Self
# python demo_denoising.py --input sem_speckle.tif --output denoised.tif
```

#### 方案 B：Log + N2V / PN2V

```python
import numpy as np

# 乘性 → 加性轉換
log_image = np.log1p(image)
log_norm = (log_image - log_image.min()) / (log_image.max() - log_image.min())
# 接續 N2V 或 PN2V 流程（與 shot noise 方案 B/A 相同架構）
# 推論後：denoised = np.expm1(denoised_log)
```

#### 方案 C：SAR-BM3D（無需訓練基準）

```python
import bm3d
import numpy as np

def sar_bm3d(image, sigma=0.1):
    log_img = np.log(image + 1e-8)
    denoised_log = bm3d.bm3d(log_img, sigma_psd=sigma,
                              stage_arg=bm3d.BM3DStages.ALL_STAGES)
    return np.exp(denoised_log)
```

---

## 6. 完整程式碼

### 計數率估計（判斷是否為 shot noise 主導）

```python
import numpy as np
import tifffile

def diagnose_noise_type(image, flat_region=None):
    """
    診斷影像噪聲類型。
    Poisson（shot）特性：方差 ≈ 均值（在平坦區域）
    Speckle 特性：方差 ≈ 均值²（方差係數 CV ≈ 常數）
    """
    img = image.astype(np.float64)

    if flat_region is None:
        # 自動選取最平坦的 10% 區域（標準差最小的 patch）
        h, w = img.shape
        ps = 32
        min_std = np.inf
        for i in range(0, h - ps, ps):
            for j in range(0, w - ps, ps):
                patch = img[i:i+ps, j:j+ps]
                if patch.std() < min_std:
                    min_std = patch.std()
                    flat_region = np.s_[i:i+ps, j:j+ps]

    roi = img[flat_region]
    mean_val = roi.mean()
    var_val  = roi.var()
    cv       = np.sqrt(var_val) / (mean_val + 1e-8)   # 變異係數

    print(f"平坦區域均值：{mean_val:.2f}")
    print(f"平坦區域方差：{var_val:.2f}")
    print(f"方差 / 均值  ：{var_val / (mean_val + 1e-8):.3f}  （Poisson 理論值 ≈ 1）")
    print(f"變異係數 CV  ：{cv:.3f}  （Speckle 理論值 ≈ 1/√L，通常 0.1–0.5）")

    if 0.5 < (var_val / mean_val) < 2.0:
        print("→ 診斷：主要為 Shot Noise（Poisson）")
    elif cv > 0.3:
        print("→ 診斷：主要為 Speckle（乘性）")
    else:
        print("→ 診斷：混合型噪聲，建議用 PN2V 或廣義 Anscombe")

    return {"mean": mean_val, "var": var_val,
            "var_over_mean": var_val / mean_val, "cv": cv}

# 使用範例
image = tifffile.imread("sem_image.tif").astype(np.float32)
stats = diagnose_noise_type(image)
```

### 去噪效果評估

```python
import numpy as np

def evaluate_denoising(original, denoised, flat_region=None):
    """
    同時計算 Shot Noise 和 Speckle 的評估指標。
    """
    if flat_region is None:
        flat_region = np.s_[50:150, 50:150]   # 預設同質背景區

    roi_orig     = original[flat_region].astype(np.float64)
    roi_denoised = denoised[flat_region].astype(np.float64)

    # --- Shot noise 指標 ---
    # SNR 提升：方差降低比例
    snr_improvement = 10 * np.log10(roi_orig.var() / (roi_denoised.var() + 1e-8))

    # --- Speckle 指標 ---
    # ENL（Equivalent Number of Looks）：越大越好
    enl_orig     = roi_orig.mean()**2     / (roi_orig.var()     + 1e-8)
    enl_denoised = roi_denoised.mean()**2 / (roi_denoised.var() + 1e-8)

    # --- 通用指標 ---
    # SSIM（需要 skimage）
    try:
        from skimage.metrics import structural_similarity as ssim
        ssim_val = ssim(original.astype(np.float32),
                        denoised.astype(np.float32),
                        data_range=original.max() - original.min())
    except ImportError:
        ssim_val = None

    print(f"SNR 提升（同質區）：+{snr_improvement:.2f} dB")
    print(f"ENL 原始影像：{enl_orig:.2f}")
    print(f"ENL 去噪結果：{enl_denoised:.2f}（提升 {enl_denoised/enl_orig:.1f}x）")
    if ssim_val is not None:
        print(f"SSIM：{ssim_val:.4f}")

    return {
        "snr_improvement_dB": snr_improvement,
        "enl_original": enl_orig,
        "enl_denoised": enl_denoised,
        "ssim": ssim_val,
    }
```

### 視覺化比較

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_comparison(original, results_dict, title="去噪結果比較"):
    """
    results_dict: {"方法名稱": denoised_array}
    """
    n = len(results_dict) + 1
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))

    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("原始影像")
    axes[0].axis("off")

    for ax, (name, img) in zip(axes[1:], results_dict.items()):
        ax.imshow(img, cmap="gray")
        ax.set_title(name)
        ax.axis("off")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig("denoising_comparison.png", dpi=150)
    plt.show()

# 使用範例
visualize_comparison(
    original=image,
    results_dict={
        "PN2V（shot noise）": denoised_pn2v,
        "Anscombe + N2V":    denoised_anscombe,
        "Self2Self（speckle）": denoised_s2s,
        "Log + N2V（speckle）": denoised_log_n2v,
    }
)
```

---

## 7. 方法選擇決策樹

```
Step 1：先用 diagnose_noise_type() 確認噪聲類型
  │
  ├─ Var/Mean ≈ 1，均勻顆粒感
  │    → Shot Noise（Poisson）
  │    │
  │    ├─ 單張影像，要求最佳效果
  │    │    └─► PN2V（Probabilistic N2V）
  │    │
  │    ├─ 單張影像，快速驗證
  │    │    └─► Anscombe + N2V（改三行）
  │    │
  │    ├─ 單張影像，最省事
  │    │    └─► N2V 直接用（shot noise 滿足像素獨立假設）
  │    │
  │    └─ 無需訓練，快速基準
  │         └─► VST + BM3D
  │
  ├─ CV > 0.3，有 grain pattern，強區噪聲也強
  │    → Speckle（乘性 Gamma）
  │    │
  │    ├─ 單張影像，grain 明顯
  │    │    └─► Self2Self
  │    │
  │    ├─ 多張影像或快速驗證
  │    │    └─► Log + N2V / PN2V
  │    │
  │    ├─ 有掃描條紋 + speckle
  │    │    └─► Log + Structured N2V
  │    │
  │    └─ 無需訓練，快速基準
  │         └─► SAR-BM3D 或 Lee Filter
  │
  └─ 兩者特徵都有（混合型）
       → 廣義 Anscombe（GAT）+ PN2V
         或 Self2Self（對噪聲分佈無假設）
```

---

## 8. 效能評估指標

| 指標 | 適合場景 | 計算方式 | 說明 |
|---|---|---|---|
| **SNR 提升（dB）** | Shot noise 首選 | `10·log₁₀(σ²_before / σ²_after)` | 同質區方差降低量 |
| **ENL** | Speckle 首選 | `μ² / σ²`（同質區）| 越大表示 speckle 越少 |
| **SSIM** | 通用 | skimage.metrics | 結構相似性，需要參考影像 |
| **PSNR** | 有 ground truth 時 | `10·log₁₀(MAX² / MSE)` | 需要乾淨參考 |
| **邊緣保留指數** | 通用 | Sobel 算子比較 | 評估細節是否過度平滑 |

---

## 9. 綜合比較總表

### 噪聲特性總覽

| 特性 | Shot Noise | Speckle |
|---|---|---|
| 來源 | 電子計數量子統計 | 相干波干涉 |
| 數學模型 | `y ~ Poisson(x)` | `y = x · n`，`n ~ Gamma(L, 1/L)` |
| 加性 / 乘性 | 加性 Poisson | 乘性 Gamma |
| 空間相關性 | **無（像素獨立）** | **有（grain pattern）** |
| 訊號相依性 | `σ² = x`（弱相依）| `σ² = x²/L`（強相依）|
| 高計數極限 | 趨近高斯 | 不趨近高斯 |

### N2N 家族適用性總覽

| 方法 | Shot Noise | Speckle | 單張可用 |
|---|---|---|---|
| N2N（原版）| ✅ 期望等價成立 | ✅ 同上 | ❌ 需配對 |
| **N2V / N2V2** | ✅ **直接使用** | ⚠️ 弱 speckle 可試 | ✅ |
| **PN2V** | ✅ **最推薦** | ✅ 支援任意分佈 | ✅ |
| R2R（原版）| ❌ 僅高斯 | ❌ 僅高斯 | ✅ |
| GR2R | ✅ 延伸至 Poisson | ✅ 延伸至 Gamma | ✅ |
| Self2Self | ✅ 可用（N2V 更佳）| ✅ **首選** | ✅ |
| Anscombe + N2V | ✅ **最快上手** | — | ✅ |
| Log + N2V | — | ✅ 標準橋接 | ✅ |
| VST + BM3D | ✅ 快速基準 | ❌ | ✅ |
| SAR-BM3D | ❌ | ✅ 快速基準 | ✅ |

### 推薦方法排序

| 排名 | Shot Noise | Speckle |
|---|---|---|
| 🥇 最推薦 | PN2V | Self2Self |
| 🥈 最快上手 | Anscombe + N2V | Log + N2V |
| 🥉 最省事 | N2V 直接用 | Log + PN2V |
| 🔧 無需訓練 | VST + BM3D | SAR-BM3D |

### 核心差異一句話

> **Shot noise 對 N2N 家族天生友善**：像素獨立性使 N2V 直接成立，Anscombe 轉換讓所有高斯去噪器可用，唯一的工作是處理好 Poisson 分佈的非高斯性。
>
> **Speckle 需要策略性橋接**：空間相關性破壞了 blind-spot 假設，乘性結構需要 log 轉換或 GR2R，強 speckle 下 Self2Self 是更可靠的選擇。

---

## 10. GitHub 資源

### Shot Noise 相關

| 儲存庫 | 說明 |
|---|---|
| [juglab/pn2v](https://github.com/juglab/pn2v) | Probabilistic N2V 官方實作 |
| [juglab/PPN2V](https://github.com/juglab/PPN2V) | Fully Unsupervised PN2V（GMM 噪聲模型）|
| [CAREamics/careamics](https://github.com/CAREamics/careamics) | 整合 N2V / PN2V / HDN 的 PyTorch 框架 |
| SHINE（Science Advances 2024）| TEM/SEM 專用電子顯微鏡去噪，論文可在 DOI: 10.1126/sciadv.ads5552 找到 |

### Speckle 相關

| 儲存庫 | 說明 |
|---|---|
| [scut-mingqinchen/Self2Self](https://github.com/scut-mingqinchen/Self2Self) | Self2Self with Dropout 官方實作 |
| [JK-the-Ko/Self2SelfPlus](https://github.com/JK-the-Ko/Self2SelfPlus) | Self2Self+ 改進版 |
| [wooseoklee4/AP-BSN](https://github.com/wooseoklee4/AP-BSN) | 真實相機空間相關噪聲 BSN |

### 綜合比較工具

| 儲存庫 | 說明 |
|---|---|
| [simfei/denoising](https://github.com/simfei/denoising) | CARE、DnCNN、N2N、N2V、PN2V 同框比較 |
| [juglab/n2v](https://github.com/juglab/n2v) | N2V + N2V2 官方實作（TensorFlow）|

---

*本文件整合自 N2N、N2V、PN2V 原始論文、Anscombe 轉換理論與 GR2R 廣義框架的分析，適用於 SEM 及一般科學影像的去噪方法選擇。*

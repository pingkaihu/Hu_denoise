# Speckle Noise 處理策略指南
### 基於 Noise2Noise 家族的理論分析與實作建議

---

## 目錄

1. [Speckle 噪聲的物理本質](#1-speckle-噪聲的物理本質)
2. [核心挑戰：為何 Speckle 特別困難](#2-核心挑戰為何-speckle-特別困難)
3. [N2N 家族相容性逐一分析](#3-n2n-家族相容性逐一分析)
4. [橋接策略：對數域轉換](#4-橋接策略對數域轉換)
5. [單張影像的處理方案](#5-單張影像的處理方案)
6. [方法選擇決策樹](#6-方法選擇決策樹)
7. [完整程式碼](#7-完整程式碼)
8. [參數與調整指南](#8-參數與調整指南)
9. [相容性總表](#9-相容性總表)

---

## 1. Speckle 噪聲的物理本質

Speckle 並非隨機電子雜訊，而是相干成像系統（SEM、SAR、超音波、OCT）中由電磁波干涉產生的物理現象，具有特定的統計結構。

### 標準統計模型

```
觀測值：  y = x · n

其中：
  x  = 真實訊號（reflectance）
  n  ~ Gamma(L, 1/L)，均值 = 1，方差 = 1/L
  L  = 等效視數（Equivalent Number of Looks, ENL）
       L 越大 → speckle 越弱；L = 1 → 完全 developed speckle
```

### Speckle 的四個關鍵統計性質

| 性質 | 描述 | 對去噪的影響 |
|---|---|---|
| **乘性（multiplicative）** | 噪聲與訊號相乘，非相加 | 零均值假設不直接成立 |
| **訊號相依（signal-dependent）** | 強訊號區噪聲也強 | 均一 noise level 假設不成立 |
| **空間相關（spatially correlated）** | 鄰近像素噪聲不獨立，形成 grain | Blind-spot 的像素獨立假設受挑戰 |
| **非高斯（non-Gaussian）** | Gamma / Rayleigh 分佈 | MSE loss 不再是最優估計子 |

### 從 SEM 的角度

SEM 的 speckle 主要來自低能電子束下的泊松計數噪聲與二次電子訊號波動，其統計性質近似 Gamma 分佈。空間相關性強度取決於加速電壓與影像放大倍率：

- **低倍率、高電壓**：空間相關性弱 → N2V 系列有效
- **高倍率、低電壓**：空間相關性強 → 需要 GR2R 或 Self2Self

---

## 2. 核心挑戰：為何 Speckle 特別困難

N2N 家族的各演算法都建立在一個或多個理論假設上，而 speckle 同時違反了其中幾個：

### 假設衝突對照表

| N2N 假設 | 數學表達 | Speckle 的現實 | 是否成立 |
|---|---|---|---|
| 零均值噪聲 | `E[n] = 0` | `E[n] = 1`（乘性）| ✗ |
| 期望等價 | `E[y\|x] = x` | `E[y\|x] = x · E[n] = x` | ✓ |
| 像素空間獨立 | `p(nᵢ, nⱼ) = p(nᵢ)p(nⱼ)` | Grain pattern 使鄰近像素相關 | ✗ |
| 高斯分佈 | `n ~ N(0, σ²)` | `n ~ Gamma(L, 1/L)` | ✗ |
| 噪聲強度均一 | `σ` 為常數 | `σ ∝ x`（signal-dependent）| ✗ |

**關鍵洞察：** 雖然「零均值」假設不成立，但「期望等價性 `E[y|x] = x`」仍然成立，這是 N2N 家族可以延伸至 speckle 的理論基礎。

---

## 3. N2N 家族相容性逐一分析

### 3.1 原始 Noise2Noise（N2N）

**理論狀態：△ 有條件成立**

期望等價性 `E[y|x] = x` 對 speckle 成立，N2N 的損失函數在期望值上等價於有乾淨影像的監督訓練。N2N 原論文本身也展示了對乘性 Bernoulli 噪聲的處理結果。

**根本限制：** SEM 幾乎不可能對同一場景取得兩次**獨立**的含噪觀測。若樣本不穩定（漂移、污染、電子束損傷），二次觀測的訊號本身已改變，N2N 的等價性亦不再成立。

---

### 3.2 Noise2Void / N2V2（Blind-Spot）

**理論狀態：✗ 假設破裂（強 speckle 時）**

N2V 的核心假設是「噪聲像素間空間獨立」，blind-spot 才能防止網路直接預測噪聲值。Speckle grain 的空間相關性直接使此假設失效：鄰近像素的噪聲資訊可以互相洩漏，網路可能學到預測 grain pattern 而非訊號。

**例外情況：** 當 SEM speckle 的空間相關長度（correlation length）遠小於 blind-spot 的遮蔽範圍時，N2V 仍可獲得合理效果（如高能量電子束、低放大倍率）。文獻亦記錄了 N2V 在聲納影像中對 speckle 效果尚可的案例。

**N2V2 的改進：** 使用 BlurPool 取代 MaxPool、去除 skip connection，主要解決棋盤格假影，並未針對 speckle 的空間相關性問題做根本修正。

---

### 3.3 Noisier2Noise

**理論狀態：✗ 不直接適用；✓ 轉換後可用**

原始 Noisier2Noise 僅針對加性噪聲推導，乘性 speckle 直接套用無理論保證。

**修正路徑（文獻驗證）：** 透過 Anscombe transform 或 log 轉換，將乘性 speckle 轉為加性高斯噪聲後，Noisier2Noise 的理論等價性恢復成立。此方法已在 OCT speckle 去噪中獲得實驗驗證。

---

### 3.4 R2R（Recorrupted-to-Recorrupted）

**理論狀態：✗ 原版限高斯；✓ GR2R 延伸至 Gamma**

原始 R2R 的理論等價性僅在高斯噪聲下被嚴格證明，直接用於 speckle 缺乏保證。

**GR2R 的突破：** Generalized R2R 將 recorruption 機制延伸至 exponential family 分佈，涵蓋 Poisson 與 Gamma 分佈。由於 speckle 正服從 Gamma 分佈，GR2R 在理論上是 N2N 家族中對 speckle 最嚴謹的直接解。

**使用條件：** 需要預先知道或估計噪聲分佈的參數（Gamma 的 shape parameter L）。

---

### 3.5 Neighbor2Neighbor

**理論狀態：✗ 空間獨立假設不成立**

Neighbor2Neighbor 透過鄰域下採樣生成偽配對，要求噪聲是「pixel-wise independent 且無偏差（`E[y|x] = x`）」的。空間相關的 speckle grain 直接違反此條件，鄰近像素的採樣之間存在噪聲依賴，導致訓練目標偏差。

---

### 3.6 Self2Self（Dropout）

**理論狀態：✓ 對噪聲類型不做假設**

Self2Self 透過 Bernoulli 採樣與 dropout ensemble，不假設特定噪聲分佈，理論上適用於任意噪聲，包括 speckle。文獻記錄了其在 OCT speckle（S2Snet）上的成功應用。

**適用條件：** 單張影像即可訓練，無需配對，是單張 SEM speckle 去噪的首選深度學習方案。

**限制：** 訓練時間較長（30–60 分鐘），且對強空間相關 speckle 效果有時不穩定。

---

## 4. 橋接策略：對數域轉換

這是連接所有 N2N 家族方法與 speckle 的核心工程手段，數學上稱為「homomorphic filtering」。

### 數學推導

```
原始乘性模型：
  y = x · n        (n ~ Gamma，均值 = 1)

取自然對數：
  log(y) = log(x) + log(n)

令：
  ỹ = log(y)       ← 觀測值（含噪）
  x̃ = log(x)       ← 真實訊號
  ñ = log(n)       ← 轉換後的噪聲

性質分析：
  - ñ = log(n) 的分佈趨近正態（中心極限定理，n 接近 1 時）
  - E[ñ] ≈ 0                    ← 零均值假設 ✓
  - ỹ = x̃ + ñ                   ← 加性模型 ✓
  - log-Gamma 的空間相關性弱於原始 Gamma   ← 空間獨立假設改善 ✓
```

### 轉換後的適用性改變

| 方法 | 轉換前 | 轉換後 |
|---|---|---|
| N2N | △ 有配對才可 | ✓ 配對問題不變，但噪聲更接近高斯 |
| N2V | ✗ | △ 弱 speckle 下有效 |
| R2R | ✗ | ✓ 可用 |
| Noisier2Noise | ✗ | ✓ 可用（文獻驗證） |
| GR2R | ✓（直接）| ✓（等效，不需轉換）|
| Self2Self | ✓（直接）| ✓（效果可能更好）|

### 實作注意事項

```python
# 轉換前必須確保：
# 1. 影像值域為正數（speckle 影像通常滿足）
# 2. 避免 log(0)，使用 log1p 或加小常數

log_image = np.log1p(image)          # 數值穩定版本
# 或
log_image = np.log(image + 1e-8)     # 避免真正的零值

# 推論後必須反轉換：
denoised = np.expm1(denoised_log)    # 對應 log1p 的反操作
# 或
denoised = np.exp(denoised_log)      # 對應 log + 1e-8 的反操作
```

---

## 5. 單張影像的處理方案

在最嚴苛的單張影像限制下，依效果與實作難度排序：

### 方案 A：Self2Self（首選深度學習方案）

**原理：** 對單張影像生成 Bernoulli 採樣實例對，以 dropout ensemble 降低預測方差，無需假設噪聲分佈。

```bash
# 安裝與執行
git clone https://github.com/scut-mingqinchen/Self2Self
cd Self2Self
conda install pytorch torchvision
python demo_denoising.py --input your_sem.tif --output denoised.tif
```

**關鍵參數：**

| 參數 | 建議值 | 說明 |
|---|---|---|
| `dropout_rate` | 0.3 | Bernoulli 採樣比例，影響 ensemble 多樣性 |
| `n_epochs` | 2000–5000 | 單張圖需要較多迭代 |
| `n_predictions` | 50–100 | 推論時的 ensemble 次數，越多越穩定 |

**訓練時間：** GPU 約 30–60 分鐘，推論約幾秒。

---

### 方案 B：Log 轉換 + N2V（現有環境最快）

若已有 CAREamics 環境，改兩行即可：

```python
import numpy as np
import tifffile
from careamics import CAREamist
from careamics.config import create_n2v_configuration

# 載入影像
image = tifffile.imread("sem_speckle.tif").astype(np.float32)
image = (image - image.min()) / (image.max() - image.min() + 1e-8)

# === 關鍵修改：對數轉換 ===
log_image = np.log1p(image)
log_norm = (log_image - log_image.min()) / (log_image.max() - log_image.min())
# ==========================

# N2V 訓練（與一般流程完全相同）
config = create_n2v_configuration(
    experiment_name="sem_speckle_n2v",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=128,
    num_epochs=150,
)

careamist = CAREamist(source=config)
train_data = log_norm[np.newaxis, ...]
careamist.train(train_source=train_data, val_percentage=0.1)

# 推論
denoised_log = careamist.predict(
    source=train_data, data_type="array", axes="YX",
    tile_size=[256, 256], tile_overlap=[48, 48],
)
denoised_log = np.squeeze(denoised_log)

# === 關鍵修改：反轉換 ===
denoised = np.expm1(denoised_log)
# =======================

tifffile.imwrite("denoised_speckle.tif", denoised)
```

---

### 方案 C：SAR-BM3D 或 Lee Filter（無需訓練基準）

快速驗證去噪潛力，用於比較深度學習方法的效益：

```python
import bm3d
import numpy as np
from scipy.ndimage import uniform_filter

# === SAR-BM3D（對數域 BM3D）===
log_img = np.log(image + 1e-8)
sigma_est = 0.1   # 依 speckle 強度調整（0.05–0.2）
denoised_log = bm3d.bm3d(log_img, sigma_psd=sigma_est,
                          stage_arg=bm3d.BM3DStages.ALL_STAGES)
denoised_bm3d = np.exp(denoised_log)

# === Lee Filter（speckle 專用傳統濾波器）===
def lee_filter(img, size=7):
    img_mean = uniform_filter(img.astype(np.float64), size)
    img_sq_mean = uniform_filter(img.astype(np.float64)**2, size)
    img_var = img_sq_mean - img_mean**2
    overall_var = np.var(img)
    weight = img_var / (img_var + overall_var + 1e-8)
    return img_mean + weight * (img - img_mean)

denoised_lee = lee_filter(image, size=7)
```

---

### 方案 D：GR2R（理論最嚴謹，需估計 Gamma 參數）

若能從影像中的同質區域估計 speckle 參數，GR2R 提供對 Gamma 分佈最直接的理論保證：

```python
import numpy as np

def estimate_ENL(image, roi_slice):
    """
    估計等效視數 L（Gamma 分佈的 shape parameter）
    從影像的同質背景區域估計

    roi_slice: 例如 np.s_[50:100, 50:100]
    """
    roi = image[roi_slice].astype(np.float64)
    mean_val = roi.mean()
    var_val = roi.var()
    L = (mean_val ** 2) / (var_val + 1e-8)
    print(f"估計 ENL = {L:.2f}（L 越大，speckle 越弱）")
    return L

L = estimate_ENL(image, np.s_[50:100, 50:100])

# 依 L 生成 Gamma 分佈的 recorruption noise（GR2R 核心）
def gamma_recorrupt(image, L, alpha=0.5):
    """
    在 Gamma 噪聲模型下生成再腐化影像
    alpha 控制再加噪強度，通常設為 0.3–0.7
    """
    shape = alpha * L
    scale = image / (shape + 1e-8)
    corrupted = np.random.gamma(shape=shape, scale=scale)
    return corrupted

# 用 corrupted 對作為訓練配對（替代乾淨影像）
corrupted_A = gamma_recorrupt(image, L, alpha=0.4)
corrupted_B = gamma_recorrupt(image, L, alpha=0.4)
# → 送入任意 N2N 架構訓練
```

---

## 6. 方法選擇決策樹

```
我的 SEM speckle 是什麼情況？
  │
  ├─ 只有 1 張影像
  │    │
  │    ├─ 先用 Lee Filter 或 SAR-BM3D 評估基準（< 1 分鐘）
  │    │
  │    ├─ Speckle 是均勻顆粒感（grain 小）
  │    │    └─► Log + N2V（5–15 分鐘訓練）
  │    │
  │    ├─ Speckle 有明顯 grain pattern 或空間結構
  │    │    └─► Self2Self（30–60 分鐘訓練）
  │    │
  │    └─ 可以估計同質區域的噪聲統計
  │         └─► GR2R（需實作 Gamma recorruption）
  │
  ├─ 有 2 張以上同場景含噪影像
  │    └─► N2N 直接適用（最強的理論保證）
  │
  ├─ 有 5–10 張不同場景 SEM 影像
  │    └─► Log + N2V（多張訓練效果更佳）
  │
  └─ 有掃描條紋 + speckle 混合
       └─► Log 轉換 + Structured N2V
           （struct_n2v_axis="horizontal" 或 "vertical"）
```

---

## 7. 完整程式碼

### 7.1 視覺化比較（多方法對比）

```python
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import bm3d
from scipy.ndimage import uniform_filter

def load_sem(path):
    img = tifffile.imread(path).astype(np.float32)
    if img.ndim == 3:
        img = img @ np.array([0.2989, 0.5870, 0.1140])
    return (img - img.min()) / (img.max() - img.min() + 1e-8)

def lee_filter(img, size=7):
    img = img.astype(np.float64)
    m = uniform_filter(img, size)
    sq = uniform_filter(img**2, size)
    v = sq - m**2
    w = v / (v + np.var(img) + 1e-8)
    return m + w * (img - m)

def sar_bm3d(img, sigma=0.1):
    log_img = np.log(img + 1e-8)
    denoised_log = bm3d.bm3d(log_img, sigma_psd=sigma,
                              stage_arg=bm3d.BM3DStages.ALL_STAGES)
    return np.exp(denoised_log)

# 載入影像
image = load_sem("your_sem.tif")

# 各方法執行
result_lee  = lee_filter(image, size=7).astype(np.float32)
result_bm3d = sar_bm3d(image, sigma=0.1)

# 視覺化
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
titles = ["原始含噪", "Lee Filter", "SAR-BM3D",
          "Self2Self / Log+N2V\n（執行後填入）"]
images = [image, result_lee, result_bm3d, image]  # 最後一欄替換為你的深度學習結果

for ax, img, title in zip(axes, images, titles):
    ax.imshow(img, cmap="gray")
    ax.set_title(title, fontsize=10)
    ax.axis("off")

plt.tight_layout()
plt.savefig("speckle_comparison.png", dpi=150)
plt.show()
```

### 7.2 ENL 指標計算（量化評估去噪效果）

ENL（Equivalent Number of Looks）是 speckle 去噪的標準評估指標：ENL 越高，表示同質區域越平滑，去噪效果越好。

```python
def compute_ENL(image, roi_slice):
    """
    計算同質區域的 ENL
    roi_slice: 影像中的背景/同質區域，例如 np.s_[50:100, 50:100]
    """
    roi = image[roi_slice].astype(np.float64)
    mean_val = roi.mean()
    var_val = roi.var()
    enl = (mean_val ** 2) / (var_val + 1e-8)
    return enl

roi = np.s_[50:100, 50:100]   # 依你的影像調整同質區域位置

print(f"原始影像  ENL = {compute_ENL(image, roi):.2f}")
print(f"Lee Filter ENL = {compute_ENL(result_lee, roi):.2f}")
print(f"SAR-BM3D  ENL = {compute_ENL(result_bm3d, roi):.2f}")
# 加入深度學習結果後一併比較
```

---

## 8. 參數與調整指南

### Self2Self 關鍵參數

| 參數 | 建議值 | 效果說明 |
|---|---|---|
| `dropout_rate` | `0.3` | 太低 → ensemble 多樣性不足；太高 → 訊號遺失 |
| `n_epochs` | `3000–5000` | 單張影像需要多次迭代收斂 |
| `n_predictions` | `50–100` | 推論 ensemble 次數，越多越穩定（收益遞減） |
| `lr` | `1e-4` | 標準 Adam 學習率 |

### Log + N2V 關鍵參數

| 參數 | 建議值 | 說明 |
|---|---|---|
| `patch_size` | `[64, 64]` | SEM 高解析度可試 `[128, 128]` |
| `batch_size` | `64–128` | 依 GPU 記憶體調整 |
| `num_epochs` | `150–200` | 單張影像建議加長 |
| `sigma`（BM3D）| `0.05–0.15` | 依 speckle 強度調整 |

### Sigma 估計方法

```python
def estimate_sigma_from_flat_region(image, roi_slice):
    """從平坦背景區域估計等效噪聲標準差（用於 BM3D sigma 設定）"""
    roi = image[roi_slice].astype(np.float64)
    log_roi = np.log(roi + 1e-8)
    sigma_log = log_roi.std()
    print(f"建議 BM3D sigma = {sigma_log:.3f}")
    return sigma_log

sigma = estimate_sigma_from_flat_region(image, np.s_[50:100, 50:100])
result_bm3d = sar_bm3d(image, sigma=sigma)
```

---

## 9. 相容性總表

| 方法 | 理論相容 | 工程可行 | 單張可用 | 訓練時間 | 推薦優先度 |
|---|---|---|---|---|---|
| **Self2Self** | ✓ 直接 | ✓ | ✓ | 30–60 min | ⭐⭐⭐⭐⭐ |
| **Log + N2V** | ✓ 轉換後 | ✓ | ✓ | 5–15 min | ⭐⭐⭐⭐ |
| **GR2R** | ✓ 直接（Gamma）| △ 需估計 L | ✓ | 15–30 min | ⭐⭐⭐⭐ |
| **SAR-BM3D** | △ 近似 | ✓ | ✓ | < 1 min | ⭐⭐⭐（基準）|
| **Lee Filter** | △ 統計近似 | ✓ | ✓ | 秒級 | ⭐⭐（基準）|
| **N2N（原版）** | △ 需配對 | △ SEM 難取得 | ✗ | — | — |
| **N2V（原版）** | ✗ 假設破裂 | △ 弱 speckle | ✓ | 5–15 min | ⭐⭐ |
| **R2R（原版）** | ✗ 高斯限定 | △ Log 轉換後 | ✓ | 10–20 min | ⭐⭐⭐ |
| **Noisier2Noise** | ✗ 加性限定 | △ Anscombe 後 | ✓ | 10–20 min | ⭐⭐⭐ |

### 核心結論

理論分析的三層答案：

**理論層（有條件 Yes）：** N2N 家族的期望等價性在 `E[y|x] = x` 成立時可推廣至乘性噪聲。GR2R 更進一步直接將理論延伸至 Gamma 分佈，是對 speckle 最嚴謹的直接延伸。

**工程層（有橋接方案的 Yes）：** 對數域轉換（homomorphic filtering）將乘性問題轉為加性問題，使整個 N2N 家族都可間接應用於 speckle，且有充分的文獻實驗佐證。

**實務層（部分 Yes）：** N2V 面對強空間相關 speckle 時，即使做 log 轉換，空間相關性的問題仍存在。此時應優先選擇 GR2R 或 Self2Self，而非強行套用 N2V。

---

## 參考資源

### GitHub

| 儲存庫 | 說明 |
|---|---|
| [scut-mingqinchen/Self2Self](https://github.com/scut-mingqinchen/Self2Self) | Self2Self 官方實作（PyTorch）|
| [JK-the-Ko/Self2SelfPlus](https://github.com/JK-the-Ko/Self2SelfPlus) | Self2Self+ 改進版（加入 IQA loss）|
| [CAREamics/careamics](https://github.com/CAREamics/careamics) | N2V 系列完整框架 |
| [wooseoklee4/AP-BSN](https://github.com/wooseoklee4/AP-BSN) | 真實相機噪聲 Blind-Spot（含空間相關噪聲處理）|

### 關鍵論文

| 論文 | 貢獻 |
|---|---|
| Lehtinen et al., ICML 2018 | Noise2Noise 原始論文，含乘性噪聲實驗 |
| Quan et al., CVPR 2020 | Self2Self with Dropout |
| Monroy et al., CVPR 2025 | Generalized R2R（GR2R），延伸至 Gamma 分佈 |
| Saha et al., arXiv 2022 | Noisier2Noise + Anscombe transform for OCT speckle |

---

*本文件基於理論分析與文獻回顧整理。SEM speckle 的實際效果取決於電子束條件、材料性質與影像設定，建議依照決策樹逐步驗證各方法效果。*

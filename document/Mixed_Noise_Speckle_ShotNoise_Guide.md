# Speckle + Shot Noise 混合去噪完整指南
### 理論分析、方法比較、實作策略與運算資源評估

---

## 目錄

1. [混合噪聲的物理本質](#1-混合噪聲的物理本質)
2. [為何不建議 2-step 序列處理](#2-為何不建議-2-step-序列處理)
3. [Deep Image Prior 的分析](#3-deep-image-prior-的分析)
4. [先驗與早停的概念解釋](#4-先驗與早停的概念解釋)
5. [推薦聯合處理策略](#5-推薦聯合處理策略)
6. [各方法運算資源比較](#6-各方法運算資源比較)
7. [完整程式碼](#7-完整程式碼)
8. [方法選擇決策樹](#8-方法選擇決策樹)
9. [總結對照表](#9-總結對照表)

---

## 1. 混合噪聲的物理本質

### SEM 影像的完整噪聲模型

SEM（掃描式電子顯微鏡）影像在實際拍攝中，通常同時存在兩種性質截然不同的噪聲：

```
y = Poisson(x · n_speckle) + N(0, σ²_read)

其中：
  x            = 真實訊號
  n_speckle    ~ Gamma(L, 1/L)，均值 = 1，乘性
  Poisson(·)   = shot noise，加性，計數統計
  N(0, σ²)     = 讀出噪聲，加性高斯
```

**關鍵耦合：shot noise 的強度取決於 speckle 的當前值（`x · n_speckle`），而非原始訊號 `x`。** 這個物理耦合是混合處理比序列處理更優越的根本原因。

### 兩種噪聲的對比

| 性質 | Shot Noise（散粒噪聲）| Speckle（散斑噪聲）|
|---|---|---|
| 物理來源 | 電子計數量子統計 | 相干波干涉效應 |
| 數學模型 | 加性 Poisson | 乘性 Gamma |
| 空間相關性 | **像素間獨立** | **有 grain pattern** |
| 訊號相依性 | `σ² = λ`（弱）| `σ² = x²/L`（強）|
| 高計數極限 | 趨近高斯 | 不趨近高斯 |
| N2V 假設 | ✅ 完整成立 | ⚠️ 空間獨立假設受挑戰 |

### 假設衝突總覽

| N2N 假設 | Shot Noise | Speckle | 混合時 |
|---|---|---|---|
| `E[y\|x] = x` | ✅ | ✅ | ✅ |
| 像素空間獨立 | ✅ | ❌ | ❌ |
| 加性模型 | ✅ 近似 | ❌ 乘性 | ❌ |
| 高斯分佈 | ✅ 高計數近似 | ❌ | ❌ |

---

## 2. 為何不建議 2-step 序列處理

### 直覺上的吸引力與實際的問題

2-step 的邏輯（先去 speckle，再去 shot noise）在工程上看起來合理，但存在三個結構性問題：

### 問題一：統計模型的耦合性破壞

Shot noise 的方差依賴 speckle 的當前值：

```
完整關係：  Var[n_shot | x, n_speckle] = x · n_speckle

若 Step 1（去 speckle）不完美，得到 ŷ = x · n_residual：
  Var[n_shot | ŷ]  ≠  ŷ   ←  Step 2 的噪聲模型假設已偏移

GAT 參數估計基於錯誤統計量 → PN2V 的 likelihood 函數偏移
```

這不是工程精度問題，而是**統計模型的結構性破壞**。

### 問題二：誤差傳遞是乘法性的，不是加法性的

```
設 Step 1 的訊號衰減率為 α，Step 2 為 β：

序列處理：  S_final = S₀ · (1 - α) · (1 - β)
聯合處理：  S_final = S₀ · (1 - γ)，  γ < α + β

兩次獨立去噪判斷 = 對訊號做兩次 regression to the mean
細節損失是乘法性的，不是加法性的
```

### 問題三：第一步引入的人工結構無法被第二步辨識

```
Step 1 輸出：  ŷ₁ = x + ε_noise_residual + ε_artifact

Step 2 的輸入分佈：  x + ε_noise_residual + ε_artifact + n_shot
Step 2 的假設輸入：  x + n_shot

Distribution shift → Step 2 對 shot noise 的估計偏移
                  → 無法修正 ε_artifact（假影）
```

### 2-step 例外情境

以下情況下 2-step 可以接受：

| 情境 | 可接受性 | 原因 |
|---|---|---|
| Speckle 遠強於 Shot noise | ⚠️ 尚可 | 偏移量相對微小 |
| 快速預覽，不需精確結果 | ✅ 實用 | 傳統方法速度快 |
| Step 1 設定為 under-denoise | ⚠️ 尚可 | 減少模型假設偏移 |
| GPU 不可用 | ✅ 唯一選擇 | 傳統串接仍優於不去噪 |

---

## 3. Deep Image Prior 的分析

### DIP 的核心機制

DIP（Ulyanov et al., 2018）的關鍵洞察：**CNN 的 encoder-decoder 架構本身就是自然影像的隱式先驗（implicit prior）**。

```python
# DIP 的訓練目標
θ* = argmin_θ  loss(f_θ(z), y)

# 其中：
#   z  = 固定的隨機輸入（不更新）
#   f_θ = 隨機初始化的 U-Net
#   y  = 含噪影像（唯一資料）
#   θ  = 網路參數（唯一優化對象）
```

**頻譜偏差（spectral bias）：** 網路先學低頻訊號結構，再逐漸擬合高頻噪聲。在訓練初期的輸出就是去噪結果，不需要任何外部訓練資料或噪聲模型假設。

### DIP 對混合噪聲的優勢

- 不需要指定噪聲分佈，Poisson + Gamma 混合型都能適應
- 不需要任何訓練資料（包括含噪影像的資料集）
- 對空間相關噪聲（speckle grain）有一定處理能力

### DIP 的核心問題：早停的脆弱性

標準 DIP 的 MSE loss 隱含高斯假設，但混合噪聲不是高斯。更嚴重的問題是早停：

```
訓練迭代數
  0  ──────────────────────────────► ∞

去噪品質
  │          ╭────╮
  │        ╭╯      ╰╮
  │      ╭╯          ╰─────────────
  │    ╭╯               過擬合階段
  └──────────────────────────────►
           ↑
       峰值（但你不知道在哪）

混合噪聲問題：
  - Speckle grain（中頻）與訊號頻率範圍重疊
  - 網路無法乾淨地「先學訊號，再碰噪聲」
  - 峰值更窄、更難偵測
```

### 改善 DIP：替換損失函數

理論上，若將損失函數替換為混合噪聲的聯合負對數似然，DIP 可以正確處理：

```python
def mixed_noise_loss(output, target, L=5.0, alpha=0.5):
    """
    混合 Poisson-Gamma 負對數似然損失
    alpha: Poisson（shot）成分權重
    L:     Gamma shape parameter（ENL，從影像估計）
    """
    eps = 1e-8
    output = output.clamp(min=eps)

    # Poisson 負對數似然（shot noise）
    poisson_nll = output - target * torch.log(output)

    # Gamma 負對數似然（speckle）
    gamma_nll = (L - 1) * torch.log(output) + L * target / output

    return (alpha * poisson_nll + (1 - alpha) * gamma_nll).mean()
```

### 早停自動偵測策略

```python
outputs = []
for i in range(n_iter):
    output = model(z)
    outputs.append(output.detach())

    if len(outputs) > window * 2:
        recent_var = torch.stack(outputs[-window:]).var(dim=0).mean().item()
        older_var  = torch.stack(outputs[-window*2:-window]).var(dim=0).mean().item()

        # 輸出間的 running variance 上升 → 過擬合開始
        if recent_var > older_var * 1.05:
            print(f"Early stopping at iter {i}")
            break
```

### DIP 的定位總結

| 情境 | DIP 適用性 |
|---|---|
| 完全不知道噪聲分佈 | ✅ 適合 |
| 單張影像，計算時間不限制 | ✅ 適合 |
| 需要批量處理多張影像 | ❌ 每張重新訓練，不實際 |
| Speckle grain 明顯 | ⚠️ 架構先驗處理能力有限 |
| 需要可重現結果 | ❌ 早停隨機性導致不穩定 |

---

## 4. 先驗與早停的概念解釋

### 先驗（Prior）

**日常語言：** 在看到資料之前，你已經知道的事。例如，「乾淨影像通常是平滑的、有結構的」就是一種先驗知識。

**在去噪中的角色：** 去噪問題是 ill-posed 的——無數個乾淨影像都能對應同一張含噪影像。先驗約束解空間，告訴你「哪些乾淨影像更合理」。

```
後驗（想要的）∝ 似然（噪聲模型）× 先驗（影像統計知識）

p(x | y)  ∝  p(y | x)  ·  p(x)
             ↑              ↑
         噪聲模型        影像先驗
```

**三種先驗類型：**

| 類型 | 說明 | 代表方法 |
|---|---|---|
| 手工設計 | 明確寫出假設（平滑、稀疏梯度）| BM3D、TV regularization |
| 學習先驗 | 從資料中學習「什麼是自然影像」| N2V、PN2V（訓練後） |
| 架構先驗 | CNN 結構本身的偏好 | DIP（不需訓練資料）|

**DIP 的架構先驗：** U-Net 的 encoder-decoder 結構天生偏好產生低頻、平滑的輸出，這個偏好在隨機初始化時就已存在，不需要任何訓練資料。

### 早停（Early Stopping）

**直覺解釋：** 用鉛筆臨摹模糊照片時，一開始你畫出有意義的輪廓（訊號），但如果繼續畫，你開始描膠片顆粒、相紙紋理（噪聲）。知道什麼時候停筆，就是早停。

**DIP 中的 ELTO 現象（早學習後過擬合）：**

```
「早停」的最佳時機：
  訓練太少 → 先驗太強，網路還沒學完訊號 → 結果模糊
  訓練太多 → 先驗失效，網路開始擬合噪聲 → 結果含噪
  理想點   → 先驗學到足夠訊號，但尚未被噪聲污染
```

**為何混合噪聲讓早停更難：**

- Speckle 的 grain pattern 存在於中頻段，與某些訊號的頻率範圍重疊
- 網路無法清楚地「先學完訊號、再碰噪聲」，兩者的學習曲線糾纏
- 峰值窗口更窄、出現時機因影像而異

---

## 5. 推薦聯合處理策略

> **推薦優先順序（修訂）**
>
> | 優先 | 方法 | 理由 |
> |---|---|---|
> | **1** | **PN2V 原始空間（不做 GAT）** | GMM 直接對 Poisson-Gamma 混合建模；無轉換冗餘；無低計數偏差問題 |
> | **2** | **GR2R** | 理論上對 Gamma 噪聲最直接；不需要轉換；需估計 ENL 參數 |
> | **3** | **Self2Self** | 零參數，無需估計；但 grain > 2px 時效果降低 |
> | **4** | **GAT + N2V**（降級快速方案）| 訓練快；低計數場景（背景均值 < 0.02）需跳過 |
> | 5（退場） | ~~GAT + PN2V~~ | 數學目標自相矛盾（詳見下方說明）；不再作為首選 |
>
> **核心邏輯轉變：** PN2V 的 GMM 本來就能擬合非高斯分佈，對 Poisson-Gamma 的直接建模比先用 GAT 扭曲噪聲結構後再建模更誠實。「聯合建模」的正確做法是選一個能在原始空間工作的方法，而非前置轉換後疊加另一個方法。

---

### 策略一：PN2V 原始空間（首選）

**原理：** PN2V 的高斯混合模型（GMM）直接在原始像素強度空間學習 Poisson-Gamma 混合的噪聲分佈，不需要任何前處理轉換。Blind-spot 架構確保訓練訊號不直接洩漏給自己。

```python
import numpy as np
import tifffile
from careamics import CAREamist
from careamics.config import create_n2v_configuration
from careamics.models.noise_models import GaussianMixtureNoiseModel

# 載入影像（保留原始計數空間，僅做最小歸一化）
image = tifffile.imread("sem_mixed.tif").astype(np.float32)
if image.ndim == 3:
    image = image @ np.array([0.2989, 0.5870, 0.1140])
image_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)

# 直接在原始空間建立噪聲模型（不做 GAT）
train_data = image_norm[np.newaxis, ...]
noise_model = GaussianMixtureNoiseModel.create_from_images(
    images=train_data,
    num_gaussians=3,      # 3 個高斯可覆蓋 Poisson-Gamma 的主要形態
)

config = create_n2v_configuration(
    experiment_name="sem_mixed_pn2v_raw",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=64,
    num_epochs=150,
)
careamist = CAREamist(source=config, noise_model=noise_model)
careamist.train(train_source=train_data, val_percentage=0.1)

denoised = np.squeeze(careamist.predict(
    source=train_data, data_type="array", axes="YX",
    tile_size=[256, 256], tile_overlap=[48, 48],
))
denoised = denoised * (image.max() - image.min()) + image.min()
tifffile.imwrite("denoised_pn2v_raw.tif", denoised.astype(np.float32))
```

**何時改用其他策略：**
- Speckle grain > 2px（高倍率）→ 改用 GR2R 或 Self2Self
- 需要批次遷移到 > 10 張不同條件影像 → 先做直方圖匹配再套用

---

### 策略二：GAT + PN2V（特定場景備選）

**適用場景：** Shot noise 強度遠大於 speckle（純 Poisson-Gaussian 混合，speckle 成分極弱），且 gain / σ_read 可準確估計。此時 GAT 的方差穩定化效果明顯，PN2V 可專注於 GAT 未能消除的殘餘非均勻性。

> **⚠️ 數學自洽性限制**
>
> GAT 的設計目標是把混合噪聲完全轉為方差 ≈ 1 的 AWGN。若成功，普通 N2V + L2 即為最優解，PN2V 的 GMM 無額外收益。若失敗（低訊號區），PN2V 實際上在對 **GAT 的轉換誤差**建模，而非原始物理噪聲。此組合的理論假設在正常訊號強度範圍外都會弱化。
>
> **極低計數（背景均值 < 0.02）：** 必須完全跳過 GAT，改用策略一。

```python
import numpy as np
import tifffile
from careamics import CAREamist
from careamics.config import create_n2v_configuration
from careamics.models.noise_models import GaussianMixtureNoiseModel

def generalized_anscombe(x, gain=1.0, sigma_read=0.0, mu_read=0.0):
    """廣義 Anscombe 轉換：同時處理 Poisson（shot）+ Gaussian（read noise）"""
    val = gain * x + gain**2 * (3.0/8.0) + sigma_read**2 - gain * mu_read
    return (2.0 / gain) * np.sqrt(np.maximum(val, 0))

def inverse_generalized_anscombe(z, gain=1.0, sigma_read=0.0):
    """精確無偏反轉換（Makitalo & Foi, 2013）"""
    val = (gain * z / 2.0)**2 + (3.0/8.0) * gain**2 \
          - sigma_read**2 - (1.0/8.0) * gain**2
    return np.maximum(val / gain**2, 0)

def estimate_read_noise(image, bg_slice=None):
    """從背景暗區估計讀出噪聲標準差"""
    if bg_slice is None:
        bg_slice = np.s_[0:30, 0:30]
    bg = image[bg_slice].astype(np.float64)
    return bg.std()

# ── 主流程 ──────────────────────────────────────────
image = tifffile.imread("sem_mixed.tif").astype(np.float32)
image_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)

# Step 1：估計讀出噪聲
sigma_read = estimate_read_noise(image_norm)

# Step 2：GAT 轉換（消除 shot noise 的訊號相依性）
img_gat = generalized_anscombe(image_norm, gain=1.0, sigma_read=sigma_read)

# Step 3：PN2V 訓練
train_data = img_gat[np.newaxis, ...]
noise_model = GaussianMixtureNoiseModel.create_from_images(
    images=train_data, num_gaussians=3,
)

config = create_n2v_configuration(
    experiment_name="sem_mixed_pn2v",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=64,
    num_epochs=150,
)
careamist = CAREamist(source=config, noise_model=noise_model)
careamist.train(train_source=train_data, val_percentage=0.1)

# Step 4：推論
denoised_gat = np.squeeze(careamist.predict(
    source=train_data, data_type="array", axes="YX",
    tile_size=[256, 256], tile_overlap=[48, 48],
))

# Step 5：反轉換
denoised = inverse_generalized_anscombe(
    denoised_gat * (img_gat.max() - img_gat.min()) + img_gat.min(),
    gain=1.0, sigma_read=sigma_read
)
denoised = denoised * (image.max() - image.min()) + image.min()
tifffile.imwrite("denoised_gat_pn2v.tif", denoised.astype(np.float32))
```

### 策略三：Self2Self（零參數，最省事）

適合完全不想估計任何噪聲參數的情況：

```bash
git clone https://github.com/scut-mingqinchen/Self2Self
cd Self2Self
python demo_denoising.py --input sem_mixed.tif --output denoised_s2s.tif
```

**Self2Self 的優勢：**
- 不假設任何噪聲分佈
- Bernoulli dropout ensemble 隱式處理所有混合成分
- 不需要早停（訓練充分後效果穩定）

**Self2Self 的代價：**
- 每張影像都重新訓練（30–90 分鐘）
- 無法批量處理

### 策略四：GAT + BM3D（快速傳統基準）

在進入深度學習方案前，用 1 分鐘評估去噪潛力：

```python
import bm3d
import numpy as np

def gat_bm3d(image, gain=1.0, sigma_read=0.0):
    """GAT + BM3D：混合噪聲的快速傳統基準"""
    val = gain * image + gain**2 * (3.0/8.0) + sigma_read**2
    img_gat = (2.0 / gain) * np.sqrt(np.maximum(val, 0))
    denoised_gat = bm3d.bm3d(img_gat, sigma_psd=1.0,
                              stage_arg=bm3d.BM3DStages.ALL_STAGES)
    val_inv = (gain * denoised_gat / 2.0)**2 + (3.0/8.0) * gain**2 \
              - sigma_read**2 - (1.0/8.0) * gain**2
    return np.maximum(val_inv / gain**2, 0)
```

### 策略選擇依據

```
Step 0：診斷計數等級
  └─ 背景區域歸一化均值 < 0.02（極低計數）？
       ├─ 是 → 直接用 PN2V 原始空間（策略一），跳過所有轉換
       └─ 否 → 繼續往下

Step 1：估計 speckle grain size
  └─ grain_px = estimate_speckle_grain_size(image)
       ├─ grain_px ≤ 2 → N2V 系列或 PN2V 都可用
       └─ grain_px > 2 → 需要 GR2R 或 Self2Self

Step 2：依資源與參數估計能力選擇
  │
  ├─ 優先首選：PN2V 原始空間（策略一）
  │    └─► 適用大多數場景，無需估計任何參數，直接建模混合噪聲
  │
  ├─ grain > 2px 且能估計 ENL？
  │    └─► GR2R（理論最直接，Gamma 分佈原生支援）
  │
  ├─ 不想估計任何參數 + grain > 2px？
  │    └─► Self2Self（零假設，但訓練時間 30–90 分鐘，無法批量）
  │
  ├─ Shot noise 遠大於 Speckle（Var/Mean ≈ 1，grain 小）？
  │    └─► GAT + N2V（快速，前提：背景均值 > 0.02）
  │
  ├─ Shot noise 遠大於 Speckle + 需要精細噪聲建模？
  │    └─► GAT + PN2V（備選策略二）
  │
  └─ 需要快速基準（< 1 分鐘）？
       └─► GAT + BM3D（策略四）
```

### 為何不用 DIP 作為首選

| 面向 | GAT + PN2V | Self2Self | DIP |
|---|---|---|---|
| 噪聲模型假設 | 混合 Poisson-Gamma | 無假設 | MSE → 隱含高斯 |
| 早停問題 | 不需要早停 | 不需要早停 | **核心瓶頸** |
| 批量推論 | ✅ 秒級 | ❌ 每張重新訓練 | ❌ 每張重新訓練 |
| 可重現性 | 好 | 中等 | 差（早停隨機） |

---

## 6. 各方法運算資源比較

以下數字以 512×512 灰階影像、RTX 3080 等級 GPU 為基準。

### 各方法資源概覽

| 方法 | 訓練時間 | 推論時間 | GPU 記憶體 | 可重用性 |
|---|---|---|---|---|
| BM3D / SAR-BM3D | 不需要訓練 | 10–60 秒（CPU）| 不需要 GPU | — |
| N2V / N2V2 | 5–15 分鐘 | 1–3 秒 | ~2–4 GB | ✅ 高 |
| PN2V / GAT+PN2V | 15–40 分鐘 | 5–15 秒 | ~3–6 GB | ✅ 高 |
| DIP（標準）| 5–20 分鐘/張 | 1–3 秒 | ~2–4 GB | ❌ 低 |
| DIP（修改損失）| 5–20 分鐘/張 | 1–3 秒 | ~2–4 GB | ❌ 低 |
| Self2Self | 30–90 分鐘/張 | 20–60 秒 | ~4–6 GB | ❌ 低 |

### 批量處理的規模效應

這是 N2V/PN2V 與 DIP/Self2Self 差距最明顯的維度：

| 方法 | N=1 張 | N=10 張 | N=100 張 |
|---|---|---|---|
| N2V（訓練一次推論多次）| ~10 分鐘 | ~12 分鐘 | ~40 分鐘 |
| DIP（每張重新訓練）| ~10 分鐘 | ~100 分鐘 | ~1000 分鐘 |
| Self2Self（每張重新訓練）| ~60 分鐘 | ~600 分鐘 | > 6000 分鐘 |

N2V/PN2V 的可重用性優勢隨影像數量增加而急劇放大：10 張時已快 8 倍，100 張時快超過 150 倍。

### 造成差異的根本原因

**N2V / PN2V 可重用：** 訓練時學習「這種 SEM 條件下噪聲的空間統計結構」，這個知識可遷移到相同條件的其他影像。推論只是一次前向傳播。

**Self2Self 不可重用：** 每張影像需要 4.5×10⁵ 步訓練，推論時還需要 50 次 dropout ensemble 取平均，沒有任何跨影像複用。

**DIP 不可重用：** 每張新影像都要讓網路從零開始重新擬合「這張特定影像」，而非「這類影像的統計」。

---

## 7. 完整程式碼

### 噪聲類型診斷

在選擇方法前，先確認影像中哪種噪聲為主：

```python
import numpy as np
import tifffile

def diagnose_noise_type(image, flat_region=None):
    """
    診斷影像噪聲類型
    Poisson（shot）特性：Var/Mean ≈ 1（在平坦區域）
    Speckle 特性：CV = std/mean ≈ 常數（0.1–0.5）
    """
    img = image.astype(np.float64)

    if flat_region is None:
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
    cv       = np.sqrt(var_val) / (mean_val + 1e-8)

    print(f"平坦區域均值：{mean_val:.2f}")
    print(f"平坦區域方差：{var_val:.2f}")
    print(f"方差 / 均值  ：{var_val / (mean_val + 1e-8):.3f}  （Poisson ≈ 1）")
    print(f"變異係數 CV  ：{cv:.3f}  （Speckle 通常 0.1–0.5）")

    if 0.5 < (var_val / mean_val) < 2.0:
        print("→ 主要為 Shot Noise（Poisson）")
    elif cv > 0.3:
        print("→ 主要為 Speckle（乘性）")
    else:
        print("→ 混合型噪聲，建議用 PN2V 原始空間（首選）或 Self2Self（無參數備選）")

image = tifffile.imread("sem_mixed.tif").astype(np.float32)
stats = diagnose_noise_type(image)
```

### ENL 估計（Speckle 強度指標）

```python
def estimate_ENL(image, roi_slice=None):
    """
    估計等效視數 L（Gamma 分佈的 shape parameter）
    ENL = mean² / variance，越大表示 speckle 越弱
    """
    if roi_slice is None:
        roi_slice = np.s_[50:100, 50:100]
    roi = image[roi_slice].astype(np.float64)
    mean_val = roi.mean()
    var_val  = roi.var()
    L = (mean_val ** 2) / (var_val + 1e-8)
    print(f"估計 ENL = {L:.2f}（L < 5 為強 speckle，L > 20 接近高斯）")
    return L
```

### 完整 GAT + PN2V 流程

```python
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from careamics import CAREamist
from careamics.config import create_n2v_configuration
from careamics.models.noise_models import GaussianMixtureNoiseModel

def full_pipeline(image_path, gain=1.0, sigma_read=None,
                  epochs=150, patch_size=64):
    """
    完整的 GAT + PN2V 混合去噪流程

    Parameters
    ----------
    image_path : str
    gain : float
        偵測器增益，不確定時設 1.0
    sigma_read : float or None
        讀出噪聲標準差，None 時自動從背景估計
    epochs : int
    patch_size : int
    """
    # 載入
    image = tifffile.imread(image_path).astype(np.float32)
    if image.ndim == 3:
        image = image @ np.array([0.2989, 0.5870, 0.1140])
    image_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)

    # 估計讀出噪聲
    if sigma_read is None:
        sigma_read = image_norm[:30, :30].std()
        print(f"估計 σ_read = {sigma_read:.4f}")

    # GAT 轉換
    val = gain * image_norm + gain**2 * (3.0/8.0) + sigma_read**2
    img_gat = (2.0 / gain) * np.sqrt(np.maximum(val, 0))
    img_gat_norm = (img_gat - img_gat.min()) / (img_gat.max() - img_gat.min() + 1e-8)
    train_data = img_gat_norm[np.newaxis, ...]

    # 噪聲模型估計
    noise_model = GaussianMixtureNoiseModel.create_from_images(
        images=train_data, num_gaussians=3,
    )

    # 訓練
    config = create_n2v_configuration(
        experiment_name="sem_mixed_pn2v",
        data_type="array", axes="YX",
        patch_size=[patch_size, patch_size],
        batch_size=64, num_epochs=epochs,
    )
    careamist = CAREamist(source=config, noise_model=noise_model)
    careamist.train(train_source=train_data, val_percentage=0.1)

    # 推論
    denoised_norm = np.squeeze(careamist.predict(
        source=train_data, data_type="array", axes="YX",
        tile_size=[256, 256], tile_overlap=[48, 48],
    ))

    # GAT 反轉換
    denoised_gat = denoised_norm * (img_gat.max() - img_gat.min()) + img_gat.min()
    val_inv = (gain * denoised_gat / 2.0)**2 + (3.0/8.0) * gain**2 \
              - sigma_read**2 - (1.0/8.0) * gain**2
    denoised = np.maximum(val_inv / gain**2, 0)
    denoised = denoised * (image.max() - image.min()) + image.min()

    # 視覺化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image, cmap='gray'); axes[0].set_title('原始含噪'); axes[0].axis('off')
    axes[1].imshow(denoised, cmap='gray'); axes[1].set_title('GAT+PN2V 去噪'); axes[1].axis('off')
    diff = np.abs(image - denoised) * 3
    axes[2].imshow(diff, cmap='hot'); axes[2].set_title('差異圖 (×3)'); axes[2].axis('off')
    plt.tight_layout()
    plt.savefig("denoising_result.png", dpi=150)
    plt.show()

    # 儲存
    tifffile.imwrite("denoised_mixed.tif", denoised.astype(np.float32))
    print("完成，已儲存至 denoised_mixed.tif")
    return denoised
```

### 去噪品質評估

```python
def evaluate_mixed_denoising(original, denoised, flat_region=None):
    """同時評估 Shot Noise 與 Speckle 的去噪效果"""
    if flat_region is None:
        flat_region = np.s_[50:150, 50:150]

    roi_orig     = original[flat_region].astype(np.float64)
    roi_denoised = denoised[flat_region].astype(np.float64)

    # SNR 提升（shot noise 指標）
    snr_db = 10 * np.log10(roi_orig.var() / (roi_denoised.var() + 1e-8))

    # ENL（speckle 指標）
    enl_orig     = roi_orig.mean()**2     / (roi_orig.var()     + 1e-8)
    enl_denoised = roi_denoised.mean()**2 / (roi_denoised.var() + 1e-8)

    print(f"SNR 提升（同質區）：+{snr_db:.2f} dB")
    print(f"ENL 原始：{enl_orig:.2f}  →  去噪後：{enl_denoised:.2f}（提升 {enl_denoised/enl_orig:.1f}x）")
```

---

## 8. 方法選擇決策樹

```
Step 1：先診斷計數等級
  └─ background_mean < 0.02（極低計數）？
       ├─ 是 → PN2V 原始空間（策略一），所有轉換方法禁用
       └─ 否 → 繼續

Step 2：診斷噪聲類型（diagnose_noise_type）
  │
  ├─ Var/Mean ≈ 1，均勻顆粒 → 主要是 Shot Noise
  │    ├─► PN2V 原始空間（首選，直接建模 Poisson）
  │    └─► 或 GAT + N2V（快速備選）
  │
  ├─ CV > 0.3，有 grain pattern → 主要是 Speckle
  │    └─► 參考《Speckle_Denoising_Strategy.md》
  │
  └─ 兩者特徵都有 → 混合型（本文件目標場景）
       │
       ├─ 首選：PN2V 原始空間（策略一）
       │    └─► 適用大多數場景，無需任何參數估計
       │
       ├─ grain > 2px（高倍率 SEM）？
       │    ├─ 能估計 ENL → GR2R
       │    └─ 不想估計  → Self2Self（30–90 分鐘/張）
       │
       ├─ 需要批量處理（> 5 張）且條件穩定？
       │    ├─► PN2V 原始空間訓練一次批量推論
       │    └─► ⚠️ 先做 check_batch_stability，偏移 > 10% 需直方圖匹配
       │
       ├─ Shot noise 明顯強於 Speckle + 需精細建模？
       │    └─► GAT + PN2V（策略二備選，非首選）
       │
       └─ 只需快速基準（< 1 分鐘）？
            └─► GAT + BM3D（策略四）
```

---

## 9. 總結對照表

### 理論適用性

| 方法 | 優先順序 | 處理混合噪聲 | 理論嚴謹性 | 早停問題 | 空間相關噪聲 |
|---|---|---|---|---|---|
| **PN2V 原始空間** | ⭐ **首選** | ✅ 直接建模 Poisson-Gamma | ✅ 最直接 | 不需要 | blind-spot 原始大小 |
| **GR2R** | ⭐⭐ | ✅ Gamma 原生支援 | ✅ 最嚴謹（Gamma）| 不需要 | 不依賴 blind-spot |
| **Self2Self** | ⭐⭐⭐ | ✅ 無假設 | ✅ 通用 | 不需要 | 隱式（grain ≤ 2px）|
| GAT + N2V | ⭐⭐⭐⭐（快速）| ✅ 近似 | ⚠️ 低計數失效 | 不需要 | blind-spot 原始大小 |
| GAT + PN2V | ⭐⭐⭐⭐⭐（特定場景）| ⚠️ 數學自相矛盾（⚠️ 見注1）| ⚠️ 有條件 | 不需要 | 擴大 blind-spot（⚠️ 見注2）|
| DIP（修改損失）| — | ✅ 修改後 | ⚠️ 中等 | **核心問題** | 架構先驗 |
| DIP（標準）| — | ⚠️ MSE 偏移 | ⚠️ 較差 | **核心問題** | 架構先驗 |
| 2-step 序列 | ❌ | ❌ 統計破壞 | ❌ | 不適用 | — |
| GAT + BM3D | 基準 | ✅ 近似 | ⚠️ 傳統方法 | 不需要 | 不處理 |

### 運算資源

| 方法 | 訓練時間/張 | 推論時間/張 | 可重用性 | GPU 記憶體 |
|---|---|---|---|---|
| GAT + BM3D | 不需要 | 10–60 秒 | — | 不需要 |
| N2V | 5–15 分鐘 | 1–3 秒 | ✅ 高 | 2–4 GB |
| **PN2V 原始空間** | 15–40 分鐘 | 5–15 秒 | ✅ 高（⚠️ 見注3）| 3–6 GB |
| GAT + PN2V | 15–40 分鐘 | 5–15 秒 | ✅ 高（⚠️ 見注3）| 3–6 GB |
| DIP | 5–20 分鐘 | 1–3 秒 | ❌ 低 | 2–4 GB |
| **Self2Self** | 30–90 分鐘 | 20–60 秒 | ❌ 低 | 4–6 GB |

### 綜合推薦

> 面對 SEM 的 speckle + shot noise 混合情境，**PN2V 原始空間（不做 GAT）** 是首選：GMM 直接對 Poisson-Gamma 混合的噪聲分佈建模，無前處理轉換的數學冗餘問題，在正常與極低計數場景下都比 GAT 組合更穩健。
>
> 若 speckle grain > 2px（高倍率 SEM），改用 **GR2R**（能估計 ENL）或 **Self2Self**（零參數）。
>
> GAT + PN2V 退為「Shot noise 遠大於 Speckle + 訊號強度正常」的特定場景備選，不再作為通用首選。
>
> 無論選哪條路，**都不要把兩種噪聲分開序列處理**——混疊的統計結構要求聯合建模。

**注1 — GAT + PN2V 的數學自洽性問題**：GAT 設計目標是把噪聲轉為 AWGN，若成功則 PN2V 的 GMM 無額外收益；若失敗（低訊號區），PN2V 在對 GAT 的轉換誤差建模，而非原始物理噪聲。兩種情況下 PN2V 都未能發揮設計用途。極低計數場景（背景均值 < 0.02）必須跳過 GAT 改用策略一。

**注2 — 擴大 blind-spot 的解析度代價**：擴大至 3×3 或 5×5 可覆蓋 speckle grain，但代價是抹平同等尺度的真實高頻結構。最小關鍵特徵 < 4px 時，改用 AP-BSN 或 N2V2，而非擴大盲區。

**注3 — 批次遷移風險**：「可重用性高」的前提是後續影像的噪聲統計與訓練影像一致。SEM 帶電效應、束流漂移可能使背景亮度偏移 > 10%，此時需先做直方圖匹配，或分批重新訓練。

---

## 參考資源

### 關鍵論文

| 論文 | 貢獻 |
|---|---|
| Ulyanov et al., CVPR 2018 | Deep Image Prior 原始論文 |
| Quan et al., CVPR 2020 | Self2Self with Dropout |
| Krull et al., Frontiers 2020 | Probabilistic Noise2Void |
| Makitalo & Foi, IEEE TIP 2013 | Generalized Anscombe 精確反轉換 |
| Wang et al., NeurIPS 2022 | Early Stopping for DIP |
| Shi et al., IJCV 2022 | Spectral Bias of DIP |
| Lequyer et al., Nature MI 2022 | Noise2Fast（Self2Self 速度比較基準）|

### GitHub

| 儲存庫 | 說明 |
|---|---|
| [scut-mingqinchen/Self2Self](https://github.com/scut-mingqinchen/Self2Self) | Self2Self 官方實作 |
| [CAREamics/careamics](https://github.com/CAREamics/careamics) | N2V、PN2V 整合框架 |
| [juglab/pn2v](https://github.com/juglab/pn2v) | PN2V 官方實作 |
| [juglab/PPN2V](https://github.com/juglab/PPN2V) | Fully Unsupervised PN2V |
| [sun-umn/Early_Stopping_for_DIP](https://github.com/sun-umn/Early_Stopping_for_DIP) | DIP 自動早停策略 |

---

*本文件整合自對 speckle + shot noise 混合去噪的完整技術討論，涵蓋理論分析、先驗與早停概念解釋、各方法的運算資源比較，以及完整的 SEM 應用程式碼。*

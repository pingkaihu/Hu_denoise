# N2V / AP-BSN 參數設定指南

**環境假設：** NVIDIA RTX 3080 (10 GB VRAM)，Windows，Python 3.12  
**適用腳本：** `denoise.py`（careamics）、`denoise_torch.py`、`denoise_tf.py`、`denoise_apbsn.py`

> **時間估計說明：** 以 RTX 3080 實測為基準，`num_workers=0`。  
> careamics 版訓練通常比 torch/tf 版快（自動萃取 patch 數量較少）。  
> 首次執行因 CUDA kernel 編譯會額外多 1–2 分鐘。  
> CPU 執行約為 GPU 的 **20–50 倍**慢。

---

## 本指南導讀

| 我的需求 | 前往章節 |
|---|---|
| 快速查詢影像大小對應的參數 | [快速對照表](#快速對照表) |
| 不確定用 N2V 還是 AP-BSN | [演算法選擇：N2V vs AP-BSN](#演算法選擇n2v-vs-ap-bsn) |
| 解決執行錯誤或效果不佳 | [常見問題與調整](#常見問題與調整) |
| 了解參數背後的機制 | [參數機制詳解](#參數機制詳解) |

---

## 快速對照表

| 影像大小 | `patch_size` | `batch_size` | `num_epochs` | `tile_size` | `tile_overlap` | `num_patches/epoch`* | 訓練時間（GPU） | 推論時間（GPU） | 總時間（GPU） | 總時間（CPU） |
|---|---|---|---|---|---|---|---|---|---|---|
| 64 × 64 | 32 | 32 | 300 | [64, 64] | [0, 0] | 500 | ~2 min | <1 s | ~2 min | ~40 min |
| 128 × 128 | 32 | 64 | 200 | [128, 128] | [0, 0] | 1 000 | ~2 min | <1 s | ~2 min | ~45 min |
| 256 × 256 | 64 | 64 | 150 | [256, 256] | [0, 0] | 1 500 | ~3 min | ~5 s | ~3 min | ~1 hr |
| 512 × 512 | 64 | 128 | 100 | [256, 256] | [48, 48] | 2 000 | ~2 min | ~10 s | ~3 min | ~1.5 hr |
| 1024 × 1024 | 64 | 128 | 100 | [256, 256] | [48, 48] | 4 000 | ~5 min | ~30 s | ~6 min | ~3 hr |
| 2048 × 2048+ | 128 | 64 | 50 | [256, 256] | [48, 48] | 8 000 | ~15 min | ~2 min | ~17 min | ~8 hr |
| CPU 執行 | 64 | 16 | 50 | [128, 128] | [32, 32] | 1 000 | ~1.5 hr | ~5 min | ~1.5 hr | — |

\* `num_patches/epoch` 僅適用於 `denoise_torch.py` / `denoise_tf.py`（`patches_per_epoch` 變數）；careamics 會自動計算。

---

## 演算法選擇：N2V vs AP-BSN

| | N2V（`denoise.py` / `denoise_torch.py`） | AP-BSN（`denoise_apbsn.py`） |
|---|---|---|
| **適用噪聲** | 像素獨立噪聲（高斯、泊松）— 標準 SEM 噪聲 | 空間相關噪聲（相機 ISP 噪聲、SEM 掃描線偽影） |
| **推論方式** | 分塊（tile）推論，大圖需設 `tile_size` | PD 縮小後整圖推論，無需 tiling |
| **速度** | 訓練快；推論需 tiling 時較慢 | 訓練相近；`avg_shifts=True` 時推論做 r² 次 pass |
| **何時選用** | 一般 SEM 影像（預設首選） | 影像有空間相關紋理或 ISP 噪聲時改用 |

---

## N2V 各尺寸參數設定

### N2V 參數選擇原則

#### patch_size 選擇

```
影像邊長 64   → patch_size 最大 32
影像邊長 128  → patch_size 最大 64（建議 32）
影像邊長 256+ → patch_size 64（標準）
影像邊長 2048+→ patch_size 128（可選）

規則：patch_size 必須為 8 的倍數，且 < 影像邊長
```

#### tile_overlap 設定原則

- 影像 ≤ 256px：`[0, 0]`（整張一次處理）
- 影像 > 256px：`[48, 48]`（約為 tile_size 的 20%）
- `tile_overlap` 必須 < `tile_size`
- 有格狀偽影時增加至 `[64, 96]`

---

### 64 × 64（極小）｜估計時間：~2 min（GPU）

N2V 的邊緣情況，效果有限，建議至少使用 128×128 以上的影像。

```python
# denoise.py（careamics）
config = create_n2v_configuration(
    patch_size=[32, 32],
    batch_size=32,
    num_epochs=300,
)
denoised = careamist.predict(..., tile_size=[64, 64], tile_overlap=[0, 0])

# denoise_torch.py / denoise_tf.py
main(patch_size=32, batch_size=32, num_epochs=300,
     tile_size=(64, 64), tile_overlap=(0, 0))
# 同時將 train_n2v() 內 patches_per_epoch 改為 500
```

**限制：**
- `patch_size` 必須 ≤ 影像邊長的一半（否則採樣位置過少），選擇原則見〈N2V 參數選擇原則〉
- `tile_size` 不得超過影像大小
- 訓練效果受限於影像資訊量

---

### 128 × 128（小）｜估計時間：~2 min（GPU）

```python
# denoise.py（careamics）
config = create_n2v_configuration(
    patch_size=[32, 32],
    batch_size=64,
    num_epochs=200,
)
denoised = careamist.predict(..., tile_size=[128, 128], tile_overlap=[0, 0])

# denoise_torch.py / denoise_tf.py
main(patch_size=32, batch_size=64, num_epochs=200,
     tile_size=(128, 128), tile_overlap=(0, 0))
# patches_per_epoch = 1000
```

---

### 256 × 256（中小）｜估計時間：~3 min（GPU）

```python
# denoise.py（careamics）
config = create_n2v_configuration(
    patch_size=[64, 64],
    batch_size=64,
    num_epochs=150,
)
denoised = careamist.predict(..., tile_size=[256, 256], tile_overlap=[0, 0])

# denoise_torch.py / denoise_tf.py
main(patch_size=64, batch_size=64, num_epochs=150,
     tile_size=(256, 256), tile_overlap=(0, 0))
# patches_per_epoch = 1500
```

---

### 512 × 512（中，預設）｜估計時間：~3 min（GPU）

目前 `test_sem.tif` 使用的尺寸，預設設定即針對此大小優化。

```python
# denoise.py（careamics）— 預設設定
config = create_n2v_configuration(
    patch_size=[64, 64],
    batch_size=128,
    num_epochs=100,
)
denoised = careamist.predict(..., tile_size=[256, 256], tile_overlap=[48, 48])

# denoise_torch.py / denoise_tf.py
main(patch_size=64, batch_size=128, num_epochs=100,
     tile_size=(256, 256), tile_overlap=(48, 48))
# patches_per_epoch = 2000
```

---

### 1024 × 1024（大）｜估計時間：~6 min（GPU）

```python
# denoise.py（careamics）
config = create_n2v_configuration(
    patch_size=[64, 64],
    batch_size=128,
    num_epochs=100,
)
denoised = careamist.predict(..., tile_size=[256, 256], tile_overlap=[48, 48])

# denoise_torch.py / denoise_tf.py
main(patch_size=64, batch_size=128, num_epochs=100,
     tile_size=(256, 256), tile_overlap=(48, 48))
# patches_per_epoch = 4000
```

---

### 2048 × 2048 以上（超大）｜估計時間：~17 min（GPU）

```python
# denoise.py（careamics）
config = create_n2v_configuration(
    patch_size=[128, 128],   # 更大的 patch 捕捉更多上下文
    batch_size=64,           # 128px patch VRAM 佔用較高，降低 batch
    num_epochs=50,
)
denoised = careamist.predict(..., tile_size=[256, 256], tile_overlap=[48, 48])

# denoise_torch.py / denoise_tf.py
main(patch_size=128, batch_size=64, num_epochs=50,
     tile_size=(256, 256), tile_overlap=(48, 48))
# patches_per_epoch = 8000
```

---

### CPU 執行（無 GPU）｜估計時間：~1.5 hr（512×512 基準）

```python
# denoise.py（careamics）
config = create_n2v_configuration(
    patch_size=[64, 64],
    batch_size=16,
    num_epochs=50,
)
denoised = careamist.predict(..., tile_size=[128, 128], tile_overlap=[32, 32])

# denoise_torch.py / denoise_tf.py
main(patch_size=64, batch_size=16, num_epochs=50,
     tile_size=(128, 128), tile_overlap=(32, 32))
# patches_per_epoch = 1000
```

---

## AP-BSN 參數設定（`denoise_apbsn.py`）

AP-BSN（CVPR 2022）與 N2V 最大的差異是先做 **PD（Pixel-Shuffle Downsampling）**：
把原圖 `(H, W)` 拆成 `r²` 個空間交錯的子格，形成 `(r², H/r, W/r)` 的 PD 域影像。  
空間相關噪聲在 PD 域會變成近似像素獨立，使 N2V 盲點訓練得以在 PD 域運作。

> **注意：** AP-BSN 的 `patch_size` 是 **PD 域尺寸**（已縮小 r 倍），非原始像素大小。  
> 例如原圖 512×512、`pd_stride=2` → PD 域 256×256，此時 `patch_size=64` 對應原圖 128×128。  
> N2V 的 patch_size 選擇原則不適用於 AP-BSN，請以本節規則為準。

### AP-BSN 核心參數

| 參數 | 預設值 | 說明 | 調整方向 |
|---|---|---|---|
| `pd_stride` | `2` | Pixel-Shuffle 步幅 `r`；決定 PD 域縮小比例與通道數（r²） | SEM → `2`；相機 sRGB 空間相關噪聲 → `5` |
| `avg_shifts` | `True` | 是否平均 r² 種 shift 對齊（AP 品質模式） | `True`：消除 PD 格狀偽影；`False`：單次推論，快 r² 倍 |
| `patch_size` | `64` | PD 域 patch 邊長（必須為 8 的倍數且 ≤ min(Hd, Wd)） | 太小上下文不足；太大吃 VRAM |
| `batch_size` | `64` | 每次梯度更新的 patch 數 | r² 通道比 N2V 多，預設用 64 而非 128；OOM 時減半 |
| `num_epochs` | `100` | 訓練 epoch 數 | 同 N2V；空間相關噪聲可增至 150 |

### `pd_stride` 對影像維度的影響

| 原圖大小 | `pd_stride` | PD 域大小 | PD 通道數 | avg_shifts 推論次數 |
|---|---|---|---|---|
| 512 × 512 | 2 | 256 × 256 | 4 | 4 次 |
| 512 × 512 | 5 | ~102 × 102 | 25 | 25 次 |
| 1024 × 1024 | 2 | 512 × 512 | 4 | 4 次 |
| 2048 × 2048 | 2 | 1024 × 1024 | 4 | 4 次 |
| 2048 × 2048 | 5 | ~409 × 409 | 25 | 25 次 |

> **無需 tiling**：PD 縮小讓整張影像可一次送入 GPU。2048×2048 影像在 r=2 時每次 pass 約佔 16 MB VRAM，r=5 時約 1.7 MB，均不會 OOM。

### AP-BSN 場景快速對照

| 使用場景 | `pd_stride` | `batch_size` | `num_epochs` | `avg_shifts` | 推論 passes |
|---|---|---|---|---|---|
| SEM 像素獨立噪聲（標準） | `2` | `64` | `100` | `True` | 4 |
| 相機 sRGB 空間相關噪聲 | `5` | `32` | `100` | `True` | 25 |
| 快速預覽 | `2` | `64` | `50` | `False` | 1 |
| GPU RAM 不足（< 6 GB） | `2` | `32` | `100` | `True` | 4 |
| CPU 執行 | `2` | `16` | `50` | `False` | 1 |

```python
# SEM 標準設定（512×512 影像，NVIDIA GPU）
main(
    input_path="test_sem.tif",
    pd_stride=2,
    patch_size=64,
    batch_size=64,
    num_epochs=100,
    avg_shifts=True,    # 4 passes，消除 PD 格狀偽影
)

# 快速預覽（不需高品質）
main(
    input_path="test_sem.tif",
    pd_stride=2,
    batch_size=64,
    num_epochs=50,
    avg_shifts=False,   # 單次推論，快 4 倍
)

# 相機 sRGB（空間相關 ISP 噪聲）
main(
    input_path="camera_image.tif",
    pd_stride=5,
    patch_size=64,
    batch_size=32,      # r²=25 通道佔 VRAM 較多
    num_epochs=100,
    avg_shifts=True,    # 25 passes
)
```

### AP-BSN 進階參數（`APBSNDataset`）

| 參數 | 預設值 | 說明 |
|---|---|---|
| `mask_ratio` | `0.006` | 每個 patch 中被遮蔽的像素比例（約 0.6%）；與 N2V 同理，一般不需調整 |
| `neighbor_radius` | `5` | 盲點替換時鄰居的最大位移；影像有明顯週期紋理時可增至 8–10 |

### AP-BSN 訓練輸出格式

輸出格式與 `denoise_torch.py` 相同，每 10 epoch 一行：

```
Device: cuda  |  Parameters: 1,844,036
PD stride r=2: 4 channels  |  PD image: 256×256  (from 512×512)
patch_size=64  batch_size=64  epochs=100
Patches/epoch: train=1800  val=200
Epoch [  1/100]  train=0.001842  val=0.001956  1.8s
Epoch [ 10/100]  train=0.000312  val=0.000334  1.6s
```

> Loss 數值比 N2V 低約 1–2 個數量級，因為 PD 域值仍在 [0, 1] 但 r² 通道平均使 MSE 較小；**只需看趨勢（是否收斂），不需對比絕對值**。

---

## GR2R 參數設定（`denoise_GR2R.py`）

GR2R（CVPR 2021）不使用盲點遮蔽，改為對同一 patch 獨立做兩次「再污染」產生訓練對：

```
y1 = patch + α·σ·ε₁   (輸入)
y2 = patch + α·σ·ε₂   (目標，獨立採樣)
Loss = MSE(f(y1), y2)  — 覆蓋所有像素
```

> `patch_size`、`batch_size`、`epochs`、`tile_size`、`tile_overlap` 的調整原則與 N2V 完全相同，
> 請直接參照前面的〈快速對照表〉與〈N2V 各尺寸參數設定〉。

### GR2R 專屬核心參數

| 參數 | 預設值 | 說明 | 調整方向 |
|---|---|---|---|
| `--alpha` | `1.0` | 再污染強度倍率：`σ_r = alpha × σ_estimated`。控制訓練信號強度與偏差的平衡。 | 見下方 alpha 選擇指南 |
| `--noise_std` | `0`（自動） | 手動指定噪聲 σ（0 = 用 Laplacian MAD 自動估計） | 有外部測量值時才手動指定 |
| `--poisson` | `False` | 改用 Poisson 再污染模式（適合散粒噪聲主導的 SEM） | 見下方噪聲模式選擇 |
| `--photon_scale` | `100` | Poisson 模式下的光子計數尺度（僅在 `--poisson` 時有效） | 見下方 photon_scale 估計 |

---

### `--alpha` 選擇指南

alpha 決定再污染噪聲的強度，是 GR2R 最關鍵的調整參數：

| alpha 值 | 效果 | 適用情境 |
|---|---|---|
| `0.5` | 輕度再污染；訓練信號較弱但偏差小 | 噪聲已很低；影像細節豐富需保留 |
| `1.0`（預設） | 再污染 σ 與原始噪聲相當；平衡點 | 大多數 SEM 影像的起點 |
| `1.5` | 較強再污染；訓練信號更穩定 | 高噪聲影像；loss 收斂不穩定時 |
| `2.0` | 強再污染；可能引入輕微模糊偏差 | 極高噪聲；噪聲估計不準確時的保守選擇 |

> **建議先以 `alpha=1.0` 執行，若輸出仍殘留可見噪聲再試 `1.5`；若輸出偏模糊則降至 `0.5`。**

---

### 噪聲模式選擇：Gaussian vs Poisson

| 情況 | 建議模式 | 原因 |
|---|---|---|
| 標準 SEM（混合噪聲） | Gaussian（預設） | MAD 估計的 σ 能涵蓋大部分加性噪聲 |
| 低電子劑量 SEM（高散粒噪聲） | `--poisson` | 散粒噪聲的方差與信號成正比，Poisson 再污染更符合物理 |
| 不確定時 | Gaussian | 先跑預設；若亮區殘留「顆粒感」強於暗區，再試 `--poisson` |

---

### `--photon_scale` 估計方法

`photon_scale` 是 Poisson 模式的尺度參數，對應影像像素值 `[0,1]` 對應的最大光子數：

```python
# 粗估方式：觀察影像最亮區域的局部標準差
# 對 Poisson 噪聲：Var(y) ≈ mean(y) / photon_scale
# 因此：photon_scale ≈ mean(bright_patch) / var(bright_patch)

import numpy as np, tifffile
img = tifffile.imread('your_sem.tif').astype(float)
img = (img - img.min()) / (img.max() - img.min())          # 正規化
bright = img[img > 0.7]                                    # 取亮區
photon_scale = bright.mean() / (bright.var() + 1e-10)
print(f"建議 photon_scale ≈ {photon_scale:.0f}")
```

| 估計值 | 物理意義 | 對應 SEM 條件 |
|---|---|---|
| 50–100 | 低光子計數 | 低劑量、高速掃描 |
| 100–300（預設 100） | 中等計數 | 一般 SEM 條件 |
| 300–1000 | 高光子計數 | 長時間積分、高劑量 |

> `photon_scale` 只影響再污染的方差大小（類似 Gaussian 模式的 alpha × σ），估計誤差在 2× 以內對結果影響有限。

---

### GR2R 場景快速對照

| 使用場景 | `alpha` | `noise_std` | `--poisson` | `photon_scale` |
|---|---|---|---|---|
| 標準 SEM，未知噪聲 | `1.0` | `0`（自動） | 否 | — |
| 細節豐富，輕噪聲 | `0.5` | `0`（自動） | 否 | — |
| 高噪聲，收斂不穩 | `1.5` | `0`（自動） | 否 | — |
| 低劑量散粒噪聲主導 | `1.0` | — | 是 | `100`（或估計值） |
| 已知儀器噪聲 σ | `1.0` | 實測值 | 否 | — |

```bash
# 標準執行（自動估計噪聲）
python denoise_GR2R.py --input data/test_sem.tif

# 輕噪聲影像，保留細節
python denoise_GR2R.py --alpha 0.5

# 高散粒噪聲
python denoise_GR2R.py --poisson --photon_scale 80

# 已知 σ，強化訓練
python denoise_GR2R.py --noise_std 0.04 --alpha 1.5
```

### GR2R 訓練輸出格式

```
Device: cuda  |  Model parameters: 1,844,036
Re-corruption mode: Gaussian(σ_r=0.04823)
patch_size=64  batch_size=128  epochs=100
Patches/epoch: train=1800  val=200

Epoch [  1/100]  train=0.003241  val=0.003389  time=1.9s
Epoch [ 10/100]  train=0.002156  val=0.002201  time=1.7s
```

> GR2R 的 loss 數值**高於** N2V，因為 target `y2` 本身含噪聲（MSE 下限 = 再污染噪聲方差 `σ_r²`），而非乾淨影像。這是正常現象，**只看趨勢是否收斂**，不需與 N2V loss 比較絕對值。
> 
> 若 loss 始終停在 `σ_r²`（即 `alpha²·σ²`）附近不繼續下降，說明網路已無法從再污染中提取更多信號，可嘗試降低 `alpha` 或增加 `epochs`。

---

## 常見問題與調整

### GPU 顯示 OOM（VRAM 不足）

依序縮小以下參數：
1. `tile_size`：[256,256] → [128,128] → [64,64]
2. `batch_size`：減半
3. `patch_size`：128 → 64

### 訓練 loss 下降緩慢

- 增加 `num_epochs`
- 提高 `num_patches/epoch`（torch/tf 版）
- 確認影像已正規化至 [0, 1]

### 影像有明顯水平 / 垂直條紋（掃描線雜訊）

在 `denoise.py` 的 `create_n2v_configuration()` 加入：
```python
struct_n2v_axis="horizontal"  # 或 "vertical"
```

### tile_overlap 設定原則

tile_overlap 設定原則請見〈[N2V 參數選擇原則](#n2v-參數選擇原則)〉。

---

## 參數機制詳解（`main()` / `train_n2v()`）

### 訓練參數

| 參數 | 預設值 | 機制 | 調整方向 |
|---|---|---|---|
| `patch_size` | 64 | 從原圖隨機裁出 N×N 小塊作訓練單位；決定網路能看到的上下文範圍。選擇原則見〈N2V 參數選擇原則〉 | 太小（<32）上下文不足；太大（>128）吃 VRAM |
| `batch_size` | 128 | 每次梯度更新同時處理的 patch 數；影響梯度穩定性與速度 | 大 batch → 穩定但吃 VRAM；OOM 時減半 |
| `num_epochs` | 100 | 訓練資料集被完整跑過的次數；結合 `patches_per_epoch=2000` 決定總更新次數 | 單張影像建議 100–200；太多可能過擬合噪聲 |
| `tile_size` | (256,256) | 推論時切塊大小；越大邊界銜接越少但 VRAM 需求越高 | OOM 時：256→128→64 |
| `tile_overlap` | (48,48) | 相鄰 tile 的重疊寬度（約 tile_size 的 20%）；防止邊界拼接痕跡。設定原則見〈N2V 參數選擇原則〉 | 有格狀偽影時增加至 64–96 |

### 進階參數（`train_n2v()` 專屬）

#### `learning_rate = 4e-4`

搭配 **Adam 優化器 + Cosine Annealing** 學習率排程：

```python
optimizer = optim.Adam(model.parameters(), lr=4e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
```

學習率隨 epoch 從 `4e-4` 平滑衰減至 `1e-6`，避免後期震盪：

```
epoch   1 → lr ≈ 4e-4  (最大，快速學習)
epoch  50 → lr ≈ 2e-4  (中段)
epoch 100 → lr ≈ 1e-6  (最小，精細收斂)
```

- `4e-4` 是 N2V 社群驗證的經驗值，一般**不需調整**
- 若 loss 震盪不收斂 → 減小至 `1e-4`；收斂過慢 → 增至 `8e-4`

#### `val_percentage = 0.1`

將 `patches_per_epoch=2000` 的 10% 劃為驗證集：

```
train patch：2000 × 0.9 = 1800  → 參與梯度更新
val patch：  2000 × 0.1 =  200  → 只做前向傳播，不更新權重
```

- 驗證集用來監控 `val_loss` 趨勢，判斷是否過擬合
- 注意：N2V 是自監督，train/val 皆來自同一張圖，`val_loss` 主要作趨勢參考

### 參數間的量化關聯

```
patches_per_epoch = 2000（固定於 train_n2v() 內）
    ├─ × val_percentage (0.1) → val:   200 patch
    └─ × (1 − 0.1)           → train: 1800 patch
                                      ÷ batch_size (128) → ~14 steps/epoch
                                      × num_epochs (100) → ~1,400 次梯度更新
                                                   learning_rate: cosine decay 4e-4 → 1e-6
```

---

## 解讀訓練輸出

執行 `denoise_torch.py` 時每 10 個 epoch 輸出一行：

```
Epoch [  1/100]  train_loss=0.337753  val_loss=0.299617  elapsed=2.1s
Epoch [ 10/100]  train_loss=0.005203  val_loss=0.008156  elapsed=1.7s
Epoch [ 20/100]  train_loss=0.003526  val_loss=0.003599  elapsed=1.7s
```

### `train_loss`

訓練集上**被遮蔽像素**的 MSE（N2V 核心：遮住一個像素，用鄰居預測它）：

```python
loss = loss_fn(pred * mask, clean_tgt * mask)  # 只算 mask 位置的誤差
```

- 數值越低 = 網路預測越準確
- Epoch 1 → 0.34：初始隨機預測；Epoch 10 → 0.005：快速收斂

### `val_loss`

驗證集上的相同 MSE，**不參與梯度更新**（`torch.no_grad()`）：

| `val_loss` vs `train_loss` | 代表 |
|---|---|
| `val ≈ train` | 正常收斂，泛化良好（如 Epoch 20） |
| `val >> train` | 過擬合（網路記住訓練 patch 的噪聲） |
| `val` 持平但 `train` 仍下降 | 可考慮提早停止訓練 |

### `elapsed`

單一 epoch 耗時（train + validate 合計）：

- 第 1 epoch 較慢（CUDA kernel 初始化），之後穩定
- 總訓練時間估算：`elapsed × num_epochs`（如 1.7s × 100 ≈ 170s ≈ 3 分鐘）

### 健康訓練的典型曲線

```
Epoch   1：loss 高（0.3+）  → 初始化，正常
Epoch  10：loss 大幅下降   → GPU 加速發揮，快速學習
Epoch  20：loss 趨穩，val ≈ train → 收斂健康
Epoch 100：loss 極低且穩定  → 訓練完成
```

若 `train_loss` 在 Epoch 50 後仍持續下降但 `val_loss` 不降，考慮減少 `num_epochs` 或降低 `learning_rate`。

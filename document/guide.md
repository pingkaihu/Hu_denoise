# SEM 影像去噪技術指南
### 基於 Noise2Noise 系列的深度學習方法

---

## 目錄

1. [技術背景](#1-技術背景)
2. [Noise2Noise 核心原理](#2-noise2noise-核心原理)
3. [延伸技術家族](#3-延伸技術家族)
4. [SEM 影像的噪聲特性](#4-sem-影像的噪聲特性)
5. [推薦方案：Noise2Void](#5-推薦方案noise2void)
6. [實作流程](#6-實作流程)
7. [完整程式碼](#7-完整程式碼)
8. [參數調整指南](#8-參數調整指南)
9. [常見問題排除](#9-常見問題排除)
10. [GitHub 資源彙整](#10-github-資源彙整)

---

## 1. 技術背景

傳統監督式影像去噪需要大量「含噪輸入 → 乾淨輸出」的配對資料，在實際應用中往往難以取得。NVIDIA 研究員於 2018 年提出的 **Noise2Noise** 顛覆了這個假設，開啟了一系列不需要乾淨影像的去噪研究方向。

### 傳統方法 vs. 自監督方法

| 方法類型 | 訓練資料需求 | 典型代表 |
|---|---|---|
| 監督式 | 含噪 + 乾淨配對影像 | DnCNN、CARE |
| Noise2Noise | 同場景兩張含噪影像 | NVlabs/noise2noise |
| 自監督 | 單張含噪影像即可 | Noise2Void、N2V2 |
| 傳統非學習 | 無需訓練 | BM3D、NLM |

---

## 2. Noise2Noise 核心原理

### 數學基礎

Noise2Noise 的關鍵洞察：只要噪聲是**零均值（zero-mean）且相互獨立**，用兩張含噪影像訓練的效果在期望值上等同於有乾淨影像。

```
E[ L(f(x + n₁), x + n₂) ] = E[ L(f(x + n₁), x) ]

其中 n₁, n₂ 為零均值獨立噪聲
```

訓練時的損失目標設為含噪影像 B，而非乾淨影像，網路在優化過程中會自然地「平均掉」隨機噪聲，收斂到相同的去噪解。

### 適用噪聲類型

- 高斯噪聲（Gaussian noise）
- 泊松噪聲（Poisson noise）
- 乘性噪聲（Bernoulli / multiplicative）
- 隨機文字遮擋（random text overlay）

---

## 3. 延伸技術家族

### 演進脈絡

```
Noise2Noise (2018)
  └─► Noise2Void / N2V2 (2019/2022)   ← 單張影像，blind-spot 遮蔽
       └─► AP-BSN (CVPR 2022)          ← 真實相機噪聲，pixel-shuffle
            └─► MM-BSN (CVPRW 2023)    ← 多遮罩，大尺度相關噪聲
            └─► CBSN (ICCV 2023)       ← Conditional blind-spot
  └─► Noise2Self (2019)                ← J-invariant 統一理論框架
  └─► R2R (2021)                       ← 再加噪生成偽配對
  └─► Diffusion Denoisers (2022+)      ← DDRM、DiffPIR、Score-based
```

### 各方法比較

| 技術 | 發表年份 | 訓練需求 | 噪聲假設 | 適合場景 |
|---|---|---|---|---|
| Noise2Noise | 2018 | 配對含噪影像 ×2 | 零均值獨立 | 天文、MRI 多次掃描 |
| Noise2Void | 2019 | 單張影像 | 像素獨立 | 顯微鏡、SEM |
| N2V2 | 2022 | 單張影像 | 像素獨立 | 改善棋盤格假影 |
| AP-BSN | 2022 | 單張影像 | 空間相關可處理 | 真實相機 sRGB |
| R2R | 2021 | 單張影像 | 加性噪聲 | 通用相機噪聲 |
| BM3D | — | 無需訓練 | 高斯 | 快速基準線驗證 |

---

## 4. SEM 影像的噪聲特性

SEM（掃描式電子顯微鏡）影像具有獨特的噪聲結構，選擇去噪方法前需先理解：

### 主要噪聲來源

- **泊松噪聲**：電子計數的統計波動（訊號相依，強度越弱噪聲越明顯）
- **高斯讀出噪聲**：偵測器電路引入的背景雜訊
- **掃描條紋（scan artifacts）**：水平或垂直方向的空間相關噪聲

### 噪聲類型與對應方法

| 噪聲類型 | 特徵 | 推薦方法 |
|---|---|---|
| 泊松 + 高斯混合 | 像素間獨立 | Noise2Void / N2V2 |
| 水平掃描條紋 | 水平方向相關 | Structured N2V（`axis="horizontal"`）|
| 垂直掃描條紋 | 垂直方向相關 | Structured N2V（`axis="vertical"`）|
| 未知類型 | — | 先用 BM3D 評估基準 |

---

## 5. 推薦方案：Noise2Void

針對 **< 10 張 SEM 影像、GPU 可用、幾秒一張**的使用情境，推薦使用 **CAREamics 框架的 Noise2Void**。

### 選擇理由

- 不需要任何配對影像，單張 SEM 即可訓練
- SEM 泊松噪聲空間獨立，完全符合 N2V 理論假設
- CAREamics 為顯微鏡影像設計，原生支援 2D 灰階
- API 極簡，訓練到推論不超過 30 行核心程式碼
- 有 GPU 時訓練約 5–15 分鐘，推論僅需幾秒

---

## 6. 實作流程

```
安裝環境
  ↓
載入 SEM 影像（float32 灰階正規化）
  ↓
（可選）BM3D 快速基準測試
  ↓
建立 N2V 設定（patch_size, epochs）
  ↓
自監督訓練 ← 約 5–15 分鐘（GPU）
  ↓
推論去噪 ← 約幾秒/張
  ↓
視覺化確認（原圖 / 去噪 / 差異圖）
  ↓
結果滿意？ → 是 → 儲存 tif
            → 否 → 調整參數或改用 Structured N2V
```

---

## 7. 完整程式碼

### 7.1 安裝

```bash
pip install careamics tifffile matplotlib numpy bm3d
```

### 7.2 載入與前處理

```python
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from pathlib import Path

def load_sem_image(path: str) -> np.ndarray:
    """載入 SEM 影像，統一為 float32 灰階 numpy array"""
    img = tifffile.imread(path).astype(np.float32)

    # 若是 RGB 轉灰階
    if img.ndim == 3 and img.shape[-1] == 3:
        img = img @ np.array([0.2989, 0.5870, 0.1140])

    # 正規化到 [0, 1]
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img

image = load_sem_image("your_sem_image.tif")
print(f"影像尺寸: {image.shape}, 範圍: [{image.min():.3f}, {image.max():.3f}]")
```

### 7.3 多張影像批次載入

```python
def load_sem_folder(folder: str) -> np.ndarray:
    """載入資料夾內所有 SEM 影像，回傳 (N, H, W) array"""
    folder = Path(folder)
    images = []

    for ext in ["*.tif", "*.tiff", "*.png"]:
        for p in sorted(folder.glob(ext)):
            img = load_sem_image(str(p))
            images.append(img)
            print(f"  載入: {p.name} -> {img.shape}")

    # 統一裁切到最小共同尺寸
    min_h = min(img.shape[0] for img in images)
    min_w = min(img.shape[1] for img in images)
    images = [img[:min_h, :min_w] for img in images]

    return np.stack(images, axis=0)  # (N, H, W)
```

### 7.4 BM3D 基準線（可選，建議先跑）

```python
import bm3d

sigma = 0.05  # 依噪聲強度調整（0.02–0.15）
denoised_bm3d = bm3d.bm3d(image, sigma_psd=sigma)
print("BM3D 完成，可先目視確認去噪程度")
```

### 7.5 N2V 訓練

```python
from careamics import CAREamist
from careamics.config import create_n2v_configuration

config = create_n2v_configuration(
    experiment_name="sem_n2v",
    data_type="array",
    axes="YX",                # 2D 灰階
    patch_size=[64, 64],      # 建議 64 或 128
    batch_size=128,
    num_epochs=100,           # 少量影像建議 100–200
)

careamist = CAREamist(source=config)

# 單張：加 batch 維度
train_data = image[np.newaxis, ...]        # (1, H, W)
# 多張：直接使用 load_sem_folder 的結果   # (N, H, W)

careamist.train(
    train_source=train_data,
    val_percentage=0.1,
)
print("訓練完成！")
```

### 7.6 推論去噪

```python
denoised = careamist.predict(
    source=train_data,
    data_type="array",
    axes="YX",
    tile_size=[256, 256],     # 大圖分 tile，避免 OOM
    tile_overlap=[48, 48],
)
denoised = np.squeeze(denoised)
```

### 7.7 視覺化與儲存

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image, cmap='gray')
axes[0].set_title('原始 SEM 影像')
axes[0].axis('off')

axes[1].imshow(denoised, cmap='gray')
axes[1].set_title('N2V 去噪結果')
axes[1].axis('off')

diff = np.abs(image - denoised) * 3  # 差異放大 3x
axes[2].imshow(diff, cmap='hot')
axes[2].set_title('差異圖 (×3)')
axes[2].axis('off')

plt.tight_layout()
plt.savefig("denoising_result.png", dpi=150)
plt.show()

tifffile.imwrite("denoised_sem.tif", denoised)
print("已儲存至 denoised_sem.tif")
```

### 7.8 Structured N2V（有掃描條紋時使用）

```python
config = create_n2v_configuration(
    experiment_name="sem_struct_n2v",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=128,
    num_epochs=100,
    struct_n2v_axis="horizontal",   # 水平條紋用 "horizontal"，垂直用 "vertical"
    struct_n2v_span=5,
)
```

---

## 8. 參數調整指南

| 情境 | `patch_size` | `batch_size` | `num_epochs` | 備註 |
|---|---|---|---|---|
| < 5 張，GPU 8GB | `[64, 64]` | `64` | `200` | 多訓練補資料不足 |
| 5–10 張，GPU 8GB | `[64, 64]` | `128` | `100` | 標準設定 |
| 高解析度（> 2048px）| `[128, 128]` | `32` | `100` | 避免 OOM |
| 只有 CPU | `[64, 64]` | `16` | `50` | 訓練較慢，約 30–60 分鐘 |

### tile_size 與 GPU 記憶體

推論時若遇到 OOM（記憶體不足），依序嘗試：

```python
tile_size=[256, 256]   # 預設，8GB GPU 通常可行
tile_size=[128, 128]   # 降低若 OOM
tile_size=[64, 64]     # CPU 推論備用
```

---

## 9. 常見問題排除

### 問題一：去噪結果出現棋盤格假影

**原因**：N2V 原版使用 ResUNet + MaxPool，可能產生棋盤格。  
**解決**：升級到 N2V2（CAREamics 已內建），或調整 `batch_size` 與 `patch_size`。

### 問題二：水平/垂直條紋噪聲無法去除

**原因**：掃描條紋是空間相關噪聲，普通 N2V 假設像素獨立，對此無效。  
**解決**：改用 Structured N2V，設定 `struct_n2v_axis` 對應條紋方向。

### 問題三：GPU OOM

**解決**：縮小 `tile_size`（推論時）或 `batch_size`（訓練時）。

### 問題四：去噪後影像過度平滑，細節消失

**原因**：`num_epochs` 過多，或 `patch_size` 過小導致欠擬合。  
**解決**：適當降低 epochs（嘗試 50–80），或增大 `patch_size` 至 `[128, 128]`。

### 問題五：訓練 loss 不收斂

**可能原因**：影像正規化範圍異常，或影像尺寸過小切不出足夠 patch。  
**解決**：確認正規化後範圍為 `[0, 1]`；影像至少需 256×256 以上才能有效切 patch。

---

## 10. GitHub 資源彙整

### 官方實作

| 儲存庫 | 說明 | 框架 |
|---|---|---|
| [NVlabs/noise2noise](https://github.com/NVlabs/noise2noise) | Noise2Noise ICML 2018 官方實作 | TensorFlow |
| [juglab/n2v](https://github.com/juglab/n2v) | Noise2Void + N2V2 官方實作 | TensorFlow |
| [CAREamics/careamics](https://github.com/CAREamics/careamics) | 整合 N2V、CARE、PN2V 的 PyTorch 框架 | PyTorch |

### Blind-Spot 系列（近年 CVPR/ICCV）

| 儲存庫 | 論文 | 特點 |
|---|---|---|
| [wooseoklee4/AP-BSN](https://github.com/wooseoklee4/AP-BSN) | CVPR 2022 | 處理真實相機空間相關噪聲 |
| [dannie125/MM-BSN](https://github.com/dannie125/MM-BSN) | CVPRW 2023 | 多遮罩策略，大尺度噪聲 |
| [jyicu/CBSN](https://github.com/jyicu/CBSN) | ICCV 2023 | Conditional blind-spot，無需後處理 |
| [YoungJooHan/SS-BSN](https://github.com/YoungJooHan/SS-BSN) | IJCAI 2023 | 加入非局部自相似注意力 |

### 方法比較與學習資源

| 儲存庫 | 說明 |
|---|---|
| [simfei/denoising](https://github.com/simfei/denoising) | 同時實作 CARE、DnCNN、N2N、N2V 等，方便橫向比較 |
| [hanyoseob/pytorch-noise2void](https://github.com/hanyoseob/pytorch-noise2void) | 清晰的 PyTorch N2V 教學實作 |
| [yu4u/noise2noise](https://github.com/yu4u/noise2noise) | Keras N2N，支援靈活的噪聲模型設定 |

### Diffusion Model 去噪彙整

| 儲存庫 | 說明 |
|---|---|
| [ChunmingHe/awesome-diffusion-models-in-low-level-vision](https://github.com/ChunmingHe/awesome-diffusion-models-in-low-level-vision) | 擴散模型在低階視覺任務的論文清單 |
| [lixinustc/Awesome-diffusion-model-for-image-processing](https://github.com/lixinustc/Awesome-diffusion-model-for-image-processing) | 持續更新的影像處理擴散模型彙整（含 2025） |

---

## 附錄：快速決策樹

```
我的 SEM 影像有什麼問題？
  │
  ├─ 均勻顆粒噪聲（無明顯條紋）
  │    └─► Noise2Void (CAREamics)  ← 首選
  │
  ├─ 水平或垂直掃描條紋
  │    └─► Structured N2V (struct_n2v_axis)
  │
  ├─ 不確定噪聲類型
  │    └─► 先跑 BM3D 基準線，目視判斷
  │
  └─ 需要與其他方法比較
       └─► simfei/denoising（多方法比較 repo）
```

---

*本文件整理自 NVIDIA Noise2Noise（ICML 2018）及後續延伸方法。  
CAREamics 框架文件：https://careamics.github.io*
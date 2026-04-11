# SEM 影像去噪版本比較

本專案包含四個 N2V（Noise2Void）去噪腳本，各自適用不同的噪聲類型或框架需求。

---

## 版本一覽

| 項目 | `denoise.py` | `denoise_torch.py` | `denoise_log_torch.py` | `denoise_tf.py` |
|---|---|---|---|---|
| **框架** | careamics (PyTorch 後端) | 純 PyTorch | 純 PyTorch | TensorFlow / Keras |
| **演算法** | N2V | N2V | **Log + N2V** | N2V |
| **目標噪聲** | 加性（高斯/泊松） | 加性（高斯/泊松） | **乘性 Speckle（Gamma）** | 加性（高斯/泊松） |
| **程式碼行數** | 103 | 478 | 500 | 421 |
| **新增依賴** | `careamics` | `torch>=2.0.0` | `torch>=2.0.0` | `tensorflow>=2.10` |
| **Python 版本** | ≥3.9 | ≥3.9 | ≥3.9 | **≤3.12** |
| **GPU 支援** | 自動（careamics） | 自動（`cuda` if available） | 自動（`cuda` if available） | 自動（TF 管理） |
| **輸出 TIF** | `denoised_sem.tif` | `denoised_sem_torch.tif` | `denoised_sem_log_torch.tif` | `denoised_sem_tf.tif` |
| **輸出 PNG** | `denoising_result.png` | `denoising_result.png` | `denoising_log_result.png` | `denoising_result.png` |

---

## Log + N2V 的原理（`denoise_log_torch.py` 專屬）

標準 N2V 假設噪聲為**加性、像素空間獨立**。SEM 的 Speckle 噪聲屬於**乘性 Gamma 分佈**，直接違反此假設。透過對數域轉換（homomorphic filtering）可橋接兩者：

```
乘性模型：  y = x · n     (n ~ Gamma，均值 = 1)
取對數：    log(y) = log(x) + log(n)
```

`log(n)` 的分佈趨近常態（零均值），使加性假設恢復成立。

### 完整流程

```
原始像素值 [img_min, img_max]
    ↓ normalize → 線性域 [0, 1]
    ↓ log1p + 再次正規化 → 對數域 [0, 1]   ← 關鍵轉換
    ↓ N2V 訓練 + 推論（架構與 denoise_torch.py 完全相同）
    ↓ 反正規化 + expm1 → 線性域 [0, 1]     ← 反轉換
    ↓ 還原原始像素值範圍
輸出 denoised_sem_log_torch.tif
```

---

## 演算法（三個標準 N2V 版本相同）

```
載入影像（float32 灰階，正規化至 [0,1]）
  ↓
從單張影像隨機抽取 64×64 patches（每 epoch 2000 個）
  ↓
盲點遮蔽（Blind-spot masking）
  · 隨機選取 ~0.6% 像素（約 24 個/patch）
  · 以鄰近像素值取代，讓網路無法辨識遮蔽位置
  ↓
4 層 UNet 訓練（1→32→64→128→256）
  · 損失：只計算被遮蔽位置的 MSE
  · 優化器：Adam + Cosine LR 衰減（初始 lr=4e-4）
  · 訓練/驗證分割：90% / 10%
  ↓
Hann 視窗分 tile 推論（256×256，重疊 48px）
  ↓
儲存 .tif 與 .png
```

`denoise_log_torch.py` 在「載入影像」後插入 `log1p` 轉換，在「推論」後插入 `expm1` 反轉換，其餘與 `denoise_torch.py` 完全相同。

---

## 執行效能（RTX 3080，test_sem.tif 512×512）

| 情境 | 每 epoch | 100 epochs 總時間 |
|---|---|---|
| `denoise_torch.py` GPU（patch=64, batch=128） | ~1.0 秒 | **~1.7 分鐘** |
| `denoise_log_torch.py` GPU（同參數） | ~1.0 秒 | **~1.7 分鐘** |
| `denoise_tf.py` GPU（同參數，估算） | ~1.2 秒 | **~2.0 分鐘** |
| `denoise.py`（careamics，同參數） | ~1.1 秒 | **~1.8 分鐘** |
| `denoise_torch.py` CPU（batch=16） | ~22.7 秒 | **~37.9 分鐘** |

> `denoise_log_torch.py` 的 log/expm1 運算時間可忽略不計，訓練速度與 `denoise_torch.py` 相同。

---

## 選用建議

| 情境 | 建議版本 |
|---|---|
| 快速使用，不想了解實作細節 | `denoise.py`（careamics） |
| 均勻顆粒感噪聲（grain 小，高能束、低倍率） | `denoise_torch.py` |
| **SEM Speckle（grain 明顯，低能束、高倍率）** | **`denoise_log_torch.py`** |
| 需要與 TF/Keras 生態系整合 | `denoise_tf.py`（需 Python ≤3.12） |
| 學習 N2V 演算法實作 | `denoise_torch.py` 或 `denoise_log_torch.py` |

### 不確定噪聲類型時的判斷步驟

1. 用 `bm3d.bm3d(image, sigma_psd=0.05)` 快速評估加性去噪基準
2. 觀察原始影像：若噪聲強度隨亮度增加（亮區噪聲也強），為乘性 speckle → 選 `denoise_log_torch.py`
3. 若噪聲均勻分布 → 選 `denoise_torch.py`

---

## 已知限制

- **`denoise_tf.py`**：TensorFlow 不支援 Python 3.14，需 Python ≤3.12 的環境。
- **所有版本**：預設 `patch_size=64` 需影像至少 64×64；`tile_size=256` 需影像至少 256×256。
- **標準 N2V（`denoise.py`、`denoise_torch.py`、`denoise_tf.py`）**：像素噪聲須空間獨立（SEM 泊松/高斯噪聲符合）。若有水平/垂直掃描條紋，需改用 Structured N2V（見 `guide.md`）。
- **`denoise_log_torch.py`**：log 轉換改善弱至中等 speckle；強空間相關 speckle（明顯 grain pattern）的根本解是 Self2Self 或 GR2R（見 `speckle_denoising_strategy.md`）。

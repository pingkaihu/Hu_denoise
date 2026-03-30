# SEM 影像去噪版本比較

本專案包含三個功能相同的 N2V（Noise2Void）去噪腳本，各自使用不同的深度學習框架。

---

## 版本一覽

| 項目 | `denoise.py` | `denoise_torch.py` | `denoise_tf.py` |
|---|---|---|---|
| **框架** | careamics (PyTorch 後端) | 純 PyTorch | TensorFlow / Keras |
| **程式碼行數** | 103 | 464 | 421 |
| **新增依賴** | `careamics` | `torch>=2.0.0` | `tensorflow>=2.10` |
| **Python 版本** | ≥3.9 | ≥3.9 | **≤3.12**（不支援 3.14） |
| **GPU 支援** | 自動（careamics 管理） | 自動（`cuda` if available） | 自動（TF 管理） |

---

## 演算法（三版完全相同）

所有版本皆實作相同的 N2V 流程：

```
載入影像（float32 灰階，正規化至 [0,1]）
  ↓
從單張影像隨機抽取 64×64 patches（每 epoch 2000 個）
  ↓
盲點遮蔽（Blind-spot masking）
  · 隨機選取 ~0.6% 像素（約 24 個/patch）
  · 以鄰近像素值取代（非零值），讓網路無法辨識遮蔽位置
  ↓
4 層 ResUNet 訓練（1→32→64→128→256）
  · 損失：只計算被遮蔽位置的 MSE
  · 優化器：Adam + Cosine LR 衰減（初始 lr=4e-4）
  · 訓練/驗證分割：90% / 10%
  ↓
Hann 視窗分 tile 推論（256×256，重疊 48px）
  ↓
儲存 denoised_sem.tif 與 denoising_result.png
```

---

## 程式碼長度差異原因

`denoise.py` 只有 103 行，是因為 careamics 在背後封裝了所有實作。`denoise_torch.py` 和 `denoise_tf.py` 將同樣的邏輯攤開寫出：

| 元件 | `denoise.py` | `denoise_torch.py` | `denoise_tf.py` |
|---|---|---|---|
| UNet 架構 | 0（隱藏在 careamics） | ~80 行 | ~60 行 |
| 資料集 + 遮蔽 | 0 | ~90 行 | ~70 行 |
| 訓練迴圈 | 0 | ~65 行 | ~65 行 |
| Tile 推論 | 0 | ~55 行 | ~55 行 |
| 使用者介面 | ~103 行 | ~40 行 | ~40 行 |

---

## 框架實作差異

### 模型定義

| | `denoise_torch.py` | `denoise_tf.py` |
|---|---|---|
| **API 風格** | `nn.Module` 子類別 | Keras Functional API |
| **Tensor 格式** | `(B, C, H, W)` channels-first | `(B, H, W, C)` channels-last |
| **BatchNorm** | `nn.BatchNorm2d` | `layers.BatchNormalization()` |
| **Upsample** | `nn.Upsample(bilinear) + Conv2d(1×1)` | `layers.UpSampling2D(bilinear) + Conv2D(1×1)` |

### 資料管線

| | `denoise_torch.py` | `denoise_tf.py` |
|---|---|---|
| **抽象** | `torch.utils.data.Dataset` + `DataLoader` | `tf.data.Dataset.from_generator()` |
| **多執行緒** | `num_workers=0`（Windows 限制） | `prefetch(AUTOTUNE)` |
| **洗牌** | `DataLoader(shuffle=True)` | generator 隨機取樣，內建洗牌效果 |

### 訓練迴圈

| | `denoise_torch.py` | `denoise_tf.py` |
|---|---|---|
| **梯度計算** | 隱含（`loss.backward()`） | 顯式（`tf.GradientTape`） |
| **圖模式加速** | `model.train()` / `model.eval()` | `@tf.function` 編譯為靜態圖 |
| **LR 排程** | `CosineAnnealingLR(T_max=epochs)` | `CosineDecay(decay_steps=total_steps)` |
| **GPU 指定** | `model.to(device)`，tensor `.to(device)` | 全自動，無需顯式搬移 |

### Hann 視窗

| | `denoise_torch.py` | `denoise_tf.py` |
|---|---|---|
| **產生方式** | `torch.hann_window(N, periodic=False)` | `np.hanning(N)` |
| **數值等價** | 相同（對稱視窗，兩端為零） | 相同 |

---

## 執行效能（RTX 3080，test_sem.tif 512×512）

| 情境 | 每 epoch | 100 epochs 總時間 |
|---|---|---|
| `denoise_torch.py` GPU（patch=64, batch=128） | ~1.0 秒 | **~1.7 分鐘** |
| `denoise_tf.py` GPU（同參數，估算） | ~1.2 秒 | **~2.0 分鐘** |
| `denoise.py`（careamics，同參數） | ~1.1 秒 | **~1.8 分鐘** |
| `denoise_torch.py` CPU（batch=16） | ~22.7 秒 | **~37.9 分鐘** |

> TF 數據為估算值（環境不支援 Python 3.14，無法直接量測）。三版訓練時間差異 < 15%，可視為相等。

---

## 選用建議

| 情境 | 建議版本 |
|---|---|
| 快速使用，不想了解實作細節 | `denoise.py`（careamics） |
| Python 3.14 / 需要客製化訓練邏輯 | `denoise_torch.py` |
| 需要與 TF/Keras 生態系整合（Keras 回調、TFLite 匯出等） | `denoise_tf.py`（需 Python ≤3.12） |
| 學習 N2V 演算法實作 | `denoise_torch.py` 或 `denoise_tf.py` |

---

## 已知限制

- **`denoise_tf.py`**：TensorFlow 尚不支援 Python 3.14，需在 Python ≤3.12 的環境（建議用 conda）執行。
- **所有版本**：預設 `patch_size=64` 需影像至少 64×64；`tile_size=256` 需影像至少 256×256。
- **N2V 假設**：像素噪聲必須空間獨立（SEM 泊松/高斯噪聲符合）。若有水平/垂直掃描條紋，需改用 Structured N2V（見 `guide.md` §7.8）。

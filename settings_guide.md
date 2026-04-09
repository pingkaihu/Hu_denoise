# N2V 參數設定指南

**環境假設：** NVIDIA RTX 3080 (10 GB VRAM)，Windows，Python 3.12  
**適用腳本：** `denoise.py`（careamics）、`denoise_torch.py`、`denoise_tf.py`

> **時間估計說明：** 以 RTX 3080 實測為基準，`num_workers=0`。  
> careamics 版訓練通常比 torch/tf 版快（自動萃取 patch 數量較少）。  
> 首次執行因 CUDA kernel 編譯會額外多 1–2 分鐘。  
> CPU 執行約為 GPU 的 **20–50 倍**慢。

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

## 各尺寸詳細說明

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
- `patch_size` 必須 ≤ 影像邊長的一半（否則採樣位置過少）
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

### `tile_overlap` 設定原則

- 影像 ≤ 256px：`[0, 0]`（整張一次處理）
- 影像 > 256px：`[48, 48]`（約為 tile_size 的 20%）
- `tile_overlap` 必須 < `tile_size`

---

## patch_size 選擇原則

```
影像邊長 64   → patch_size 最大 32
影像邊長 128  → patch_size 最大 64（建議 32）
影像邊長 256+ → patch_size 64（標準）
影像邊長 2048+→ patch_size 128（可選）

規則：patch_size 必須為 8 的倍數，且 < 影像邊長
```

---

## 參數機制詳解（`main()` / `train_n2v()`）

### 訓練參數

| 參數 | 預設值 | 機制 | 調整方向 |
|---|---|---|---|
| `patch_size` | 64 | 從原圖隨機裁出 N×N 小塊作訓練單位；決定網路能看到的上下文範圍 | 太小（<32）上下文不足；太大（>128）吃 VRAM |
| `batch_size` | 128 | 每次梯度更新同時處理的 patch 數；影響梯度穩定性與速度 | 大 batch → 穩定但吃 VRAM；OOM 時減半 |
| `num_epochs` | 100 | 訓練資料集被完整跑過的次數；結合 `patches_per_epoch=2000` 決定總更新次數 | 單張影像建議 100–200；太多可能過擬合噪聲 |
| `tile_size` | (256,256) | 推論時切塊大小；越大邊界銜接越少但 VRAM 需求越高 | OOM 時：256→128→64 |
| `tile_overlap` | (48,48) | 相鄰 tile 的重疊寬度（約 tile_size 的 20%）；防止邊界拼接痕跡 | 有格狀偽影時增加至 64–96 |

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

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

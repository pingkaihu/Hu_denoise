# `denoise_N2V_GMM_multi.py` 完整講解

> 適合對象：具備基礎 Python / PyTorch 概念、但第一次接觸自監督影像去噪的學習者。

---

## 目錄

1. [背景：我們在解決什麼問題？](#1-背景我們在解決什麼問題)
2. [演算法核心：PN2V 是什麼？](#2-演算法核心pn2v-是什麼)
3. [Multi-Image 版本的擴充邏輯](#3-multi-image-版本的擴充邏輯)
4. [程式整體架構](#4-程式整體架構)
5. [第 1 節：影像載入](#5-第-1-節影像載入)
6. [第 2 節：UNet 架構](#6-第-2-節unet-架構)
7. [第 3 節：GMM 雜訊模型](#7-第-3-節gmm-雜訊模型)
8. [第 4 節：多影像資料集](#8-第-4-節多影像資料集)
9. [第 5 節：GMM 預訓練](#9-第-5-節gmm-預訓練)
10. [第 6 節：UNet + GMM 聯合訓練](#10-第-6-節unet--gmm-聯合訓練)
11. [第 7 節：分塊推論](#11-第-7-節分塊推論)
12. [第 8 節：輸出儲存](#12-第-8-節輸出儲存)
13. [第 9 節：主流程 `main()`](#13-第-9-節主流程-main)
14. [超參數選擇指引](#14-超參數選擇指引)
15. [常見問題與診斷](#15-常見問題與診斷)

---

## 1. 背景：我們在解決什麼問題？

SEM（掃描式電子顯微鏡）拍攝的影像天生含有雜訊，主要來源包含：

| 雜訊來源 | 說明 |
|---|---|
| **Poisson 散粒雜訊** | 電子數量有限，低劑量下隨機起伏大，方差正比於訊號強度 |
| **Gaussian 讀取雜訊** | 偵測電路固有的電子雜訊，方差固定 |
| **充電效應 / 非線性** | 樣品帶電或偵測器非線性，造成局部亮點或暗帶 |

傳統去噪方法（如 BM3D）假設純高斯雜訊，無法對應訊號相依型（signal-dependent）雜訊。

**更根本的困難**：我們沒有乾淨的參考影像（clean reference），只有一張（或一批）有噪聲的原始影像。

---

## 2. 演算法核心：PN2V 是什麼？

### 2.1 Noise2Void (N2V) 基礎

Noise2Void（Krull et al., 2019）的關鍵洞察：

> 如果把每個像素的值「遮住」，讓神經網路只用周圍像素來預測這個被遮住的像素，那麼訓練目標自然就是「乾淨訊號」——因為雜訊是像素獨立的（pixel-independent），鄰居的雜訊無法幫你預測中心像素的雜訊，所以網路只能學到真實訊號。

這個技巧稱為 **blind-spot masking**（盲點遮罩）。

訓練資料來自**單張噪聲影像**本身，不需要任何乾淨影像配對。

### 2.2 N2V 的限制：MSE 損失假設高斯雜訊

標準 N2V 使用 MSE 損失：

```
L = || s_pred - y_obs ||²
```

這等同於假設 `p(y | s)` 是方差固定的高斯分布。  
但 SEM 的 Poisson 雜訊中，方差隨訊號變化，MSE 損失會低估暗區雜訊、高估亮區雜訊，導致去噪不均勻。

### 2.3 Probabilistic N2V (PN2V) 的解法

PN2V（Krull et al., 2020）將損失函數換成：

```
L_i = -log p(y_obs_i | s_pred_i)
```

其中 `p(y | s)` 是一個**可學習的雜訊模型**，能描述「訊號為 s 時，觀測值 y 的條件機率分布」。

這個雜訊模型不再假設固定高斯，而是用高斯混合模型（GMM）來近似任意形狀的雜訊分布。

**直觀理解**：MSE 只問「預測值有多接近觀測值」；NLL 則問「在你學到的雜訊分布下，這個觀測值有多『合理』」——後者能適應訊號相依型雜訊。

---

## 3. Multi-Image 版本的擴充邏輯

`denoise_N2V_GMM.py` 只處理單張影像；`denoise_N2V_GMM_multi.py` 的改動：

| 功能 | 單影像版 | 多影像版 |
|---|---|---|
| 訓練資料來源 | 單張影像的 patches | 所有影像的 patches，均勻抽取 |
| GMM 訓練資料 | 單張影像的像素對 | **所有影像像素對的聯集** |
| UNet | 一個 | 一個（共用） |
| GMM | 一個 | 一個（共用，假設影像雜訊分布相同） |
| 輸出 | 單張去噪影像 | 每張影像各自的去噪結果 |
| 儲存 / 載入 | 無 | `--save_model` / `--load_model` |

**多影像 GMM 的優勢**：當訓練像素對從 N 張影像聚合，低訊號區域的樣本量顯著增加，GMM 能更精確地估計暗區的方差。

**使用前提**：所有影像必須在**相同的 SEM 設定**（加速電壓、倍率、劑量）下拍攝，否則雜訊分布不同，共用 GMM 會造成誤差。

---

## 4. 程式整體架構

```
denoise_N2V_GMM_multi.py
│
├─ 第 1 節  load_sem_image()        影像載入與正規化
│           find_images()           掃描目錄
│
├─ 第 2 節  DoubleConvBlock          UNet 基本卷積單元
│           N2VUNet                 4 層 Encoder-Decoder UNet
│
├─ 第 3 節  GMMNoiseModel            GMM 雜訊模型（可學習參數）
│           ├─ log_prob()           計算 log p(y|s)
│           ├─ nll_loss()           負對數似然損失
│           └─ plot_noise_model()   視覺化診斷
│
├─ 第 4 節  MultiImagePN2VDataset    多影像 Blind-Spot Dataset
│
├─ 第 5 節  _build_neighbor_proxy()  4 鄰居均值作為訊號代理
│           pretrain_gmm_multi()    GMM 預訓練（聚合所有影像）
│
├─ 第 6 節  train_pn2v_multi()      UNet + GMM 聯合訓練（NLL 損失）
│
├─ 第 7 節  _compute_padding()      Padding 計算工具
│           predict_tiled()         分塊推論 + Hann 視窗混合
│
├─ 第 8 節  save_outputs()          輸出 TIF + 比較圖 PNG
│
└─ 第 9 節  main()                  CLI 入口，串接以上所有步驟
```

---

## 5. 第 1 節：影像載入

```python
def load_sem_image(path: str) -> Tuple[np.ndarray, float, float]:
```

**設計邏輯**：

1. 用 `tifffile.imread` 讀取（支援 `.tif` / `.tiff`；`.png` 也可讀）。
2. 若影像是 RGB 或 RGBA，轉為灰階：

   ```python
   img = img @ np.array([0.2989, 0.5870, 0.1140])
   ```

   這是 ITU-R BT.601 標準的亮度係數，等同於 `cv2.cvtColor(..., COLOR_RGB2GRAY)` 的結果。

3. 正規化到 `[0, 1]`（float32）：

   ```python
   img = (img - img_min) / (img_max - img_min + 1e-8)
   ```

   `1e-8` 防止純黑影像除以零。

4. **同時回傳原始範圍 `(img_min, img_max)`**，讓最後輸出時能還原到原始像素值域。

---

## 6. 第 2 節：UNet 架構

### DoubleConvBlock

```
Conv2d(3×3) → BatchNorm → LeakyReLU(0.1)
Conv2d(3×3) → BatchNorm → LeakyReLU(0.1)
```

- `bias=False`：BatchNorm 本身有偏移參數，Conv 的 bias 多餘，去掉可減少參數。
- `LeakyReLU(0.1)`：負值區域保留 10% 梯度，避免 ReLU 的 dying neuron 問題。

### N2VUNet

```
輸入 (1, H, W)
    │
  enc1 → enc2 → enc3 → enc4      Encoder（每層後接 MaxPool2d 降採樣）
                          │
                        dec3 ← skip from enc3   Decoder（上採樣 + skip connection）
                          │
                        dec2 ← skip from enc2
                          │
                        dec1 ← skip from enc1
                          │
                        head: Conv2d(1×1)
                          │
                    輸出 (1, H, W)
```

- **4 層**：輸入必須能被 2³ = 8 整除（因為有 3 次 MaxPool）。
- **Skip connection**：`torch.cat([upsample(enc_i), dec_i], dim=1)` 把淺層細節直接傳到解碼器，避免瓶頸層遺失空間資訊。
- **1×1 head**：最後用 1×1 卷積把特徵圖壓回 1 channel，輸出預測的乾淨訊號 `s_pred`。

**為什麼 UNet 適合去噪？**  
去噪需要「看全局上下文」（深層特徵）又要「還原像素級細節」（淺層特徵）。UNet 的 encoder-decoder + skip 結構恰好同時滿足這兩個要求。

---

## 7. 第 3 節：GMM 雜訊模型

### 數學定義

對每個高斯分量 k（共 K 個），條件分布 `p(y | s)` 的參數為：

| 參數 | 公式 | 意義 |
|---|---|---|
| 均值 | `μ_k(s) = s + offset_k` | 以訊號為中心，可學習偏移 |
| 對數方差 | `log σ²_k(s) = a_k · s + b_k` | log-linear in signal |
| 混合權重 | `w_k = softmax(log_weights)_k` | 訊號無關，固定權重 |

混合機率：

```
p(y | s) = Σ_k  w_k · N(y ; μ_k(s), σ²_k(s))
```

### 物理直覺

- `a_k > 0`：σ 隨訊號增加（Poisson 特性）
- `a_k ≈ 0`：σ 固定（Gaussian 讀取雜訊特性）
- 兩個分量 K=2 剛好對應這兩種物理噪聲來源

### log_prob 計算

```python
def log_prob(self, y, s):
    log_w   = F.log_softmax(self.log_weights, dim=0)   # log 混合權重
    mu      = s + self.mean_offsets                     # 各分量均值
    log_var = self.var_a * s + self.var_b               # 各分量 log 方差
    var     = log_var.exp() + 1e-8                      # 方差（加 ε 防止為零）

    log_gauss = -0.5 * ((y - mu)² / var + log_var + log(2π))   # 各分量 log N

    return logsumexp(log_w + log_gauss, dim=-1)         # log Σ w_k N_k
```

使用 `logsumexp` 是為了**數值穩定性**（避免直接計算 exp 後相加導致上溢/下溢）。

### plot_noise_model()

訓練完成後會自動生成診斷圖（`noise_model_PN2V_multi.png`），包含：

- **左圖**：各分量的 σ(s) 曲線（若兩條曲線重疊 → 分量退化，需減少 K）
- **右圖**：各分量的混合權重柱狀圖

---

## 8. 第 4 節：多影像資料集

```python
class MultiImagePN2VDataset(Dataset):
```

### Blind-Spot Masking 原理

對一個 patch 中隨機選 `n_masked` 個像素：

```
原始 patch:   patch[r, c] = 真實值 y
被遮蓋後:     corrupted[r, c] = patch[鄰居位置]  ← 替換為鄰居像素值
mask[r, c] = 1                                   ← 標記這個位置要計算損失
```

關鍵：**被替換掉的位置，其真實觀測值 `y_obs` 仍然保存在 `noisy_tgt` 中**，用於計算 NLL 損失。

### 向量化遮罩實作

```python
flat_idx   = rng.choice(P * P, size=n_masked, replace=False)
rows, cols = np.unravel_index(flat_idx, (P, P))

dr = rng.integers(-rad, rad + 1, size=n_masked)   # 鄰居偏移量（行）
dc = rng.integers(-rad, rad + 1, size=n_masked)   # 鄰居偏移量（列）

# 確保偏移不為 (0, 0)（否則等於不替換）
zero_mask = (dr == 0) & (dc == 0)
# ... 修正 zero_mask 的位置 ...

nr = np.clip(rows + dr, 0, P - 1)   # 邊界裁切
nc = np.clip(cols + dc, 0, P - 1)
corrupted[rows, cols] = patch[nr, nc]
```

這裡完全沒有 Python 迴圈，全部用 NumPy 向量化操作完成，DataLoader 工作時不會成為瓶頸。

### 多影像均勻抽取

```python
img_idx = int(self.rng.integers(0, self.n_images))   # 隨機選一張影像
H, W    = self.shapes[img_idx]
r0      = int(self.rng.integers(0, H - P))           # 隨機選 patch 位置
```

每次 `__getitem__` 隨機選擇影像後再隨機選 patch，實現跨影像的均勻採樣。

---

## 9. 第 5 節：GMM 預訓練

### 為什麼需要預訓練？

GMM 和 UNet 是聯合訓練的，但訓練初期 UNet 預測 `s_pred` 很不準，若 GMM 同步從隨機初始化開始，可能陷入局部最小值。

解法：先用一個粗糙但有效的訊號代理 `s_proxy` 來預訓練 GMM，給 GMM 一個好的初始點。

### 鄰居均值作為訊號代理

```python
def _build_neighbor_proxy(image):
    kernel = [0, 0.25, 0]
              [0.25, 0, 0.25]
              [0, 0.25, 0]
    s_proxy = F.conv2d(image, kernel, padding=1)
```

用 4 個直接鄰居（上下左右）的均值近似訊號：

```
s_proxy[r, c] = (y[r-1,c] + y[r+1,c] + y[r,c-1] + y[r,c+1]) / 4
```

**為什麼鄰居均值是好的訊號代理？**  
- 雜訊是像素獨立的，平均後方差降低（中心極限定理）
- 鄰居像素的訊號分量與中心像素高度相關（影像局部平滑）
- 結合後：`s_proxy ≈ s + ε_small`，是一個低雜訊的訊號估計

### 聚合所有影像的像素對

```python
for img in images:
    y_f, s_f = _build_neighbor_proxy(img)
    y_parts.append(y_f)
    s_parts.append(s_f)

y_all = torch.cat(y_parts)   # 所有影像像素的 y 值
s_all = torch.cat(s_parts)   # 所有影像像素的 s_proxy 值
```

若有 N 張 512×512 影像，聚合後有 N × 262,144 個像素對——GMM 的統計估計將大幅改善。

### 預訓練迴圈

```python
for epoch in range(n_epochs):
    idx  = torch.randperm(N)[:batch_size]   # 每次隨機取 4096 個像素對
    loss = noise_model.nll_loss(y_all[idx], s_all[idx])
    loss.backward()
    optimizer.step()
```

這是純粹的 GMM 參數優化，UNet 不參與。

---

## 10. 第 6 節：UNet + GMM 聯合訓練

### 損失函數

```python
pred   = model(noisy_in)                          # UNet 預測 s_pred
y_obs  = noisy_tgt.squeeze(1)[mask_bool]          # 被遮蓋位置的原始觀測值
s_pred = pred.squeeze(1)[mask_bool]               # 被遮蓋位置的 UNet 預測

loss   = noise_model.nll_loss(y_obs, s_pred)      # -log p(y_obs | s_pred)
```

**為什麼只在 mask 位置計算損失？**  
- 未被遮蓋的位置：`corrupted[r,c] = patch[r,c]`，UNet 可以直接複製輸入，不需要學習
- 被遮蓋的位置：`corrupted[r,c] = 鄰居值`，UNet 必須用上下文推斷，才能接近真實 `y_obs`

### 雙學習率設計

```python
optimizer = optim.Adam([
    {'params': model.parameters(),       'lr': learning_rate},           # UNet: 4e-4
    {'params': noise_model.parameters(), 'lr': learning_rate * 0.1},    # GMM:  4e-5
])
```

**為什麼 GMM 的學習率要低？**  
初期 UNet 預測不準，`s_pred` 偏差大，若 GMM 以高學習率跟著快速調整，會過度配合 UNet 的錯誤，導致噪音模型偏移。低學習率讓 GMM 緩慢跟隨，待 UNet 穩定後再細調。

### Cosine 退火學習率

```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs, eta_min=1e-6
)
```

學習率從 `learning_rate` 以餘弦曲線平滑降至 `1e-6`。這比固定學習率更能在訓練後期精細收斂。

### patches_per_epoch 的自動計算

```python
patches_per_epoch = max(2000, 500 * len(images))
```

影像數量越多，每個 epoch 抽取的 patches 也線性增加，確保每張影像在每個 epoch 都有充分的覆蓋機率。

---

## 11. 第 7 節：分塊推論

推論時不能把整張高解析度影像一次送入 UNet（GPU 記憶體不足），所以採用**分塊推論（Tiled Inference）**。

### 問題一：tile 的接縫

若直接把相鄰 tile 的結果拼接，接縫處會有明顯邊界（因為卷積的邊界效應）。

**解法：Hann 視窗加權混合**

```python
hann_2d = np.outer(
    torch.hann_window(th).numpy(),
    torch.hann_window(tw).numpy(),
)
output_sum[r:r+th, c:c+tw] += pred * hann_2d
weight_sum[r:r+th, c:c+tw] += hann_2d
```

Hann 視窗是一個從中央到邊緣逐漸衰減為 0 的權重：

```
  0 ─── 0.5 ─── 1.0 ─── 0.5 ─── 0
```

相鄰 tile 在重疊區域互相加權平均，中央貢獻多、邊緣貢獻少，接縫自然消失。

最終：

```python
output = output_sum / np.maximum(weight_sum, 1e-8)
```

### 問題二：影像尺寸不是 tile_size 的整數倍

```python
pad_h = _compute_padding(H, th)
padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
```

用反射填充（`reflect`）延伸邊緣，推論後裁回原始尺寸 `[:H, :W]`。

### 問題三：UNet 要求輸入能被 8 整除

`_compute_padding()` 在計算 padding 時同時確保 `padded_size % 8 == 0`。

### 批次推論加速

```python
for i in range(0, total_tiles, infer_batch_size):
    batch_coords = coords[i:i + infer_batch_size]
    tiles   = [padded[r:r+th, c:c+tw] for r, c in batch_coords]
    batch_t = torch.from_numpy(np.stack(tiles)).unsqueeze(1).to(device)
    preds   = model(batch_t).squeeze(1).cpu().numpy()
```

每次將多個 tile 合成一個 batch 送 GPU，大幅提高吞吐量（比一個一個送快 5-10×）。

---

## 12. 第 8 節：輸出儲存

```python
def save_outputs(image, denoised, img_min, img_max, tif_path, png_path):
```

1. **還原像素值域**：

   ```python
   denoised_original = denoised * (img_max - img_min) + img_min
   ```

   訓練在 `[0, 1]` 空間進行，輸出時還原到原始 ADC 計數（或 uint16 範圍），保持與原始影像的直接可比性。

2. **儲存 `.tif`**：用 `tifffile.imwrite`，保留 float32 精度。

3. **儲存比較圖 `.png`**：三欄對比：

   | 欄位 | 內容 |
   |---|---|
   | Original | 原始有噪聲影像 |
   | PN2V Denoised | 去噪結果 |
   | Difference (×3) | 差值放大 3 倍（`hot` colormap 顯示雜訊分布） |

---

## 13. 第 9 節：主流程 `main()`

完整執行順序：

```
Step 1   find_images(input_dir)         → 找到所有要去噪的影像
Step 2   find_images(train_dir)         → 找到訓練影像（預設同 input_dir）
Step 3   load_sem_image() × N           → 載入訓練影像並正規化
Step 4   低計數診斷                       → 警告極低訊號影像（bg_mean < 0.02）
Step 5   N2VUNet() + GMMNoiseModel()    → 建立模型

  if --load_model:
    torch.load() → 跳過訓練，直接推論

  else:
Step 6   pretrain_gmm_multi()           → GMM 預訓練（聚合所有影像像素對）
Step 7   train_pn2v_multi()             → UNet + GMM 聯合訓練

  if --save_model:
    torch.save({model, noise_model, n_gaussians})

Step 8   load_sem_image() × M           → 載入推論影像（若 train_dir != input_dir）
Step 9   noise_model.plot_noise_model() → 輸出 GMM 診斷圖
Step 10  predict_tiled() × M            → 對每張影像分塊推論
         save_outputs()                → 儲存結果
```

### 關鍵 CLI 參數說明

| 參數 | 預設值 | 說明 |
|---|---|---|
| `--input_dir` | `.` | 要去噪的影像目錄 |
| `--train_dir` | （空）| 若指定，僅用此目錄影像訓練；`input_dir` 的影像只做推論 |
| `--n_gaussians` | `3` | GMM 分量數 K |
| `--gmm_pretrain_epochs` | `300` | GMM 預訓練輪數（可降至 100 加速） |
| `--save_model` | （空）| 儲存 checkpoint 路徑 |
| `--load_model` | （空）| 載入 checkpoint，跳過所有訓練 |
| `--tile_size` | `256` | 推論分塊大小（OOM 時降至 128 或 64） |

---

## 14. 超參數選擇指引

### GMM 分量數 K

| K | 適用場景 |
|---|---|
| 2 | 純 Poisson + Gaussian，噪聲來源明確 |
| 3 | **預設**，大多數 SEM 場景 |
| 5 | 複雜多源噪聲（充電效應 + 多偵測器），需要更多影像 |

**診斷方法**：訓練後呼叫 `noise_model.plot_noise_model()`，若兩條 σ(s) 曲線幾乎重疊，表示有分量退化（collapse），應減少 K。

### Patch Size

| 場景 | 建議值 |
|---|---|
| 標準（8GB GPU） | `64` |
| 高解析度影像（> 2048px） | `128` |
| CPU 推論 | `64`（epochs 也要降至 50） |

Patch 必須能被 8 整除（UNet 架構限制）。

---

## 15. 常見問題與診斷

### Q1：訓練 NLL 不下降

可能原因：
- GMM 預訓練輪數不足（嘗試增加 `--gmm_pretrain_epochs 500`）
- K 太大導致 GMM 退化（嘗試 `--n_gaussians 2`）
- 學習率太高（預設 `4e-4` 通常合適）

### Q2：推論 OOM（Out of Memory）

降低分塊大小：`--tile_size 128` 或 `--tile_size 64`

### Q3：不同條件的影像能放在一起訓練嗎？

**不建議**。若影像拍攝條件不同（加速電壓、倍率、樣品材質），噪聲分布會不同，共用 GMM 會平均化不同的噪聲統計，導致去噪效果下降。請改用 `denoise_N2V_GMM.py` 逐張處理。

### Q4：什麼時候用 `--train_dir`？

當你有一批「代表性」影像用於訓練，但還有更多新影像只需要推論時：

```bash
# 用 10 張代表影像訓練並儲存模型
python denoise_N2V_GMM_multi.py --input_dir ./train_imgs --output_dir ./out --save_model sem_pn2v.pt

# 用儲存的模型對 100 張新影像推論（不重新訓練）
python denoise_N2V_GMM_multi.py --input_dir ./all_imgs --output_dir ./out --load_model sem_pn2v.pt
```

### Q5：背景均值 < 0.02 的警告是什麼意思？

```
WARNING: background mean=0.008 < 0.02 (extreme low-dose SEM)
```

這是低計數診斷。極低訊號影像（極低劑量 SEM）的 Poisson 雜訊非常劇烈，此時 PN2V 直接在原始空間訓練仍然正確（與對數 / GAT 變換相比，不會對暗區引入額外偏差）——這個警告只是提醒你注意，並非需要更換方法。

---

*文件版本：2026-04-14  |  對應腳本：`denoise_N2V_GMM_multi.py`*

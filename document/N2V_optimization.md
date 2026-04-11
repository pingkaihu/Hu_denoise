# Noise2Void (N2V) PyTorch 實作優化分析與建議

這份文件針對原版的純 PyTorch 實作 `denoise_N2V.py` 進行了深度的效能與架構分析，並提出了五大維度的優化策略。這些優化方案已落實於 `denoise_N2V_test.py` 中，顯著提升了運算效能並增強了推論時的強健性與物理意義。

---

## 1. 資料處理瓶頸：CPU 遮罩生成向量化 (Vectorized Masking)

### 分析與理由
在原始版本中，N2V 演算法需要隨機把 Patch 內的某些像素替換為「隨機相鄰特徵」以產生盲點(Blind-spot)。這段邏輯是透過 Python `for` 迴圈以及 `while True` (為了防範中心 `(0, 0)` 的重疊) 逐一運算。
當這個迴圈包裝在 PyTorch 的 `Dataset.__getitem__` 裡時，每次抽取訓練樣本皆需在 CPU 端執行幾十次運算。這會造成極為嚴重的 CPU 效能瓶頸，使得 Dataloader 的供給跟不上 GPU 的訓練消化速度，結果就是 GPU 平均利用率十分低下。

### 優化建議與重要程式碼
建議將替換邏輯改用 Numpy 的向量化 (Vectorization) 操作，一口氣抹除 CPU 端的 Python 迴圈負載。

```python
# 【優化前】使用耗時的 for 與 while 迴圈
for r, c in zip(rows, cols):
    while True:
        dr = int(self.rng.choice(dr_choices))
        dc = int(self.rng.choice(dc_choices))
        if dr != 0 or dc != 0: break
    # ... 後續依序替換更新像素 ...

# ----------------------------------------------------

# 【優化後】導入 Numpy 向量化陣列操作
dr = self.rng.integers(-rad, rad + 1, size=self.n_masked)
dc = self.rng.integers(-rad, rad + 1, size=self.n_masked)
zero_mask = (dr == 0) & (dc == 0)

# 若不巧抽到 (0,0) 的自身點，則直接利用 bool mask 整批加上隨機的 -1 或 1 做位移
if np.any(zero_mask):
    dr[zero_mask] += self.rng.choice([-1, 1], size=np.sum(zero_mask))

nr = np.clip(rows + dr, 0, P - 1)
nc = np.clip(cols + dc, 0, P - 1)
corrupted[rows, cols] = patch[nr, nc]
mask[rows, cols] = 1.0
```

---

## 2. GPU 閒置問題：推論階段批次並行處理 (Batched Inference)

### 分析與理由
遇到解析度較高的大圖時，原版的拼接推論 `predict_tiled` 一次只取單獨一塊切割好的 Tile 送入神經網路，也就是 `model(tile_t).squeeze()`，這意味著推論的 Batch Size 始終為 1。對於擁有千萬個核心的現代 GPU 來說，這無法有效壓榨 CUDA 效能，不僅拖慢了推論時間，還會大幅增加 CPU 與 GPU 之間的資料傳遞 (I/O) 往返開銷。

### 優化建議與重要程式碼
建議將全數裁切好的 Tiles 在 CPU 端收集成多個批次（Batch，如每次 8 或 16 張），再透過 `torch.stack` 拼成標準的四維矩陣，一次性送給 GPU 推演。

```python
# 將原圖的切割區塊放入列表堆疊後，進行並行批次推論
for i in range(0, total_tiles, batch_size):
    batch_coords = coords[i:i+batch_size]
    # 一次性擷取多張影像塊
    tiles = [torch.from_numpy(padded_image[r0:r0+th, c0:c0+tw]) for (r0, c0) in batch_coords]
    
    # 整合成 shape: (B, 1, H, W)
    batch_t = torch.stack(tiles).unsqueeze(1).to(device)
    
    # 一次性讓 GPU 大量計算完並拉回 CPU
    preds = model(batch_t).squeeze(1).cpu().numpy()
    
    # 接著再用 for 迴圈乘上 Hann Window 合成至大圖上...
```

---

## 3. 防範過擬合錯覺 (Data Leakage) 與物理分割

### 分析與理由
原版 `train_ds` 與 `val_ds` 分別從「一模一樣的單張輸入圖像」中進行全域的隨機抽取。由於 SEM 這種實體觀測影像具有高度的空間連續性，這會造成驗證集的 Patch 特徵跟訓練集的 Patch 高度重疊 (Data Leakage)。在這種情形下，當你看見 `val_loss` 跟著下降時，它只是在反映與 `train_loss` 差不多的東西，而失去了評估 Early Stopping 或是模型「泛化能力 (Generalization)」的守門員價值。

### 優化建議與重要程式碼
在資料進入 Dataset 物件前，依據比例 (如 80%/20%) 行影像的「實體空間切割 (Physical Split)」，確保訓練點與驗證點在地理位置上不相連。

```python
# 實體切割上下半部，保障訓練與驗證集的純潔性
split_idx = int(image.shape[0] * (1 - val_percentage))
train_image = image[:split_idx, :]
val_image = image[split_idx:, :]

# 若給定的輸入影像極微小，導致預切後小於 patch_size，為防崩潰在此安插保險機制
if train_image.shape[0] < patch_size or val_image.shape[0] < patch_size:
    print("Image too small... reverting to randomization")
    train_image = image; val_image = image

train_ds = N2VDataset(train_image, ...)
val_ds   = N2VDataset(val_image, ...)
```

---

## 4. 解決極端邊界案例 (Edge Cases & Padding)

### 分析與理由
典型的 Tiled Inference 需要依賴固定區塊尺寸（如 `256×256`）和交疊步長來行走。如果使用者給了一張只有 `128×128` 的微型影像做快速預覽，或是提供了一張長寬非 $8$ 的公因數之特殊尺寸，原本的寫法會引發座標計算崩潰，抑或直接報出 Shape 分布不均的錯誤 (Assert Error)。

### 優化建議與重要程式碼
在進入推論迴圈前，引入硬性的「反射補邊 (Reflection Padding)」。若原圖長寬較小就依差額向外補足，而推論完畢後再精確剪裁掉不要的墊片。

```python
th, tw = tile_size
# 計算需要的長／寬最低限度補邊量
pad_h = max(0, th - image.shape[0])
pad_w = max(0, tw - image.shape[1])

# 無論大小，都保證需要能被 8 整除 (UNet Downsample 需求)
if pad_h == 0 and pad_w == 0:
    pad_h = (8 - image.shape[0] % 8) % 8
    pad_w = (8 - image.shape[1] % 8) % 8

# 使用 Numpy 鏡像填充 padding，降低神經網路推論時遇到"純黑邊界"的震盪影響
padded_image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')

# ... 在加了 padding 的圖上推論 ... 
# 推論完成後，將多運算或多突出的維距裁切掉，返還原汁原味的影像大小
denoised = denoised_pad[:image.shape[0], :image.shape[1]]
```

---

## 5. 無痛兼顧乘性噪聲：平滑擴增特徵 (Log Transformation)

### 分析與理由
SEM 的顯微掃描成像是充滿多種物理干擾的，特別是**散斑雜訊 (Speckle)** 以及**散粒雜訊 (Poisson)** 帶有絕對的乘積 (Multiplicative) 與訊號相依特性。單純呼叫常態分配假設為主的 N2V 並只給予 MSE Loss (均方誤差)，容易在數學估計上對「亮暗對比極強的地帶」產生推算失真與偏差。

### 優化建議與重要程式碼
無須大動刀修改神經網路，可在載入與儲存這最外圍的兩扇大門增設把關機制：`use_log_transform`，預先把乘性難題橋接為加性白噪聲 (AWGN)。

```python
def load_sem_image(path: str, use_log_transform: bool = False):
    img = tifffile.imread(path).astype(np.float32)
    img_min = float(img.min())
    
    if use_log_transform:
        # np.log1p (即 log(x+1)) 能杜絕影像值剛好為 0 的溢位錯誤，平穩過渡
        img = np.log1p(img - img_min)

    # 隨後將影像再作正常的 0 到 1 極值正規化 ...
```

```python
def save_outputs(..., use_log_transform: bool):
    # 最後，在進行反向正規化（將數值拉長回原值域）之後：
    if use_log_transform:
        # 反向指數轉換 ( exp(x)-1 )，由於避免 float 的極端意外，用 np.maximum 保底不見負
        denoised_final = np.expm1(np.maximum(0, denoised_original)) + img_min
        image_vis = np.expm1(np.maximum(0, image_original)) + img_min
```

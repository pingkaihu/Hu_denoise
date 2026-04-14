# 參數設定指南

**環境假設：** NVIDIA RTX 3080 (10 GB VRAM)，Windows，Python 3.12  
**適用腳本：** `denoise_N2V.py`、`denoise_log_N2V.py`、`denoise_N2V_multi.py`、`denoise_log_N2V_multi.py`、`denoise_PN2V.py`、`denoise_PN2V_multi.py`、`denoise_apbsn.py`、`denoise_apbsn_multi.py`、`denoise_apbsn_faithful.py`、`denoise_apbsn_faithful_multi.py`、`denoise_GR2R.py`、`denoise_GR2R_multi.py`、`denoise_DIP.py`

> **時間估計說明：** 以 RTX 3080 實測為基準，`num_workers=0`。  
> 首次執行因 CUDA kernel 編譯會額外多 1–2 分鐘。  
> CPU 執行約為 GPU 的 **20–50 倍**慢。

---

## 本指南導讀

| 我的需求 | 前往章節 |
|---|---|
| 不確定用哪支腳本 | [演算法選擇](#演算法選擇) |
| 快速查詢影像大小對應的參數（N2V / PN2V 系列） | [快速對照表](#快速對照表n2v--pn2v-系列) |
| 設定 N2V（標準 / log / 多圖） | [N2V 系列](#n2v-系列) |
| 設定 PN2V（混合噪聲 / 多圖） | [PN2V 系列](#pn2v-系列) |
| 設定 AP-BSN（空間相關噪聲 / 多圖） | [AP-BSN 系列](#ap-bsn-系列) |
| 設定 GR2R（單張） | [GR2R 參數設定](#gr2r-參數設定) |
| 設定 GR2R（多張） | [GR2R 多張](#denoise_gr2r_multipy多張) |
| 設定 Log+N2V（多張，乘性噪聲） | [Log+N2V 多張](#denoise_log_n2v_multipy多張乘性噪聲) |
| 設定 DIP（Deep Image Prior） | [DIP 參數設定](#dip-參數設定) |
| 解讀訓練中的 loss 數值 | [解讀訓練輸出](#解讀訓練輸出) |
| 解決執行錯誤或效果不佳 | [常見問題與調整](#常見問題與調整) |
| 了解參數背後的機制 | [參數機制詳解](#參數機制詳解) |

---

## 演算法選擇

### 決策流程

```
影像有水平 / 垂直掃描線條紋？
  └─ 是 → denoise_apbsn.py（空間相關噪聲；論文完整版 → denoise_apbsn_faithful.py）
  └─ 否 ↓

多張相似條件的影像？
  └─ 是，噪聲類型均勻  → denoise_N2V_multi.py
  └─ 是，噪聲混合複雜  → denoise_PN2V_multi.py
  └─ 是，乘性 / 散斑   → denoise_log_N2V_multi.py
  └─ 是，未知加性噪聲  → denoise_GR2R_multi.py
  └─ 是，掃描線問題    → denoise_apbsn_multi.py（或 denoise_apbsn_faithful_multi.py 論文完整版）
  └─ 否（單張）↓

不確定噪聲類型，或 N2V 留有格狀偽影？
  └─ 是 → denoise_DIP.py（無噪聲模型假設）
  └─ 否 ↓

乘性 / 散斑噪聲（亮暗區顆粒感差異明顯）？
  └─ 是 → denoise_log_N2V.py

混合噪聲（Poisson + Gaussian + 偵測器非線性）？
  └─ 是 → denoise_PN2V.py
  └─ 否 → denoise_N2V.py（標準 SEM 均勻噪聲）

想要無盲點遮蔽、完整感受野訓練？
  └─ 是 → denoise_GR2R.py
```

### 方法速覽

| 腳本 | 最適噪聲類型 | 訓練資料需求 | 特色 |
|---|---|---|---|
| `denoise_N2V.py` | 像素獨立噪聲（高斯 / 泊松） | 單張 | 標準首選 |
| `denoise_log_N2V.py` | 乘性 / 散斑噪聲 | 單張 | log 變換後轉加性噪聲 |
| `denoise_N2V_multi.py` | 像素獨立，多圖 | 多張 | 共用模型，支援 save/load |
| `denoise_log_N2V_multi.py` | 乘性 / 散斑，多圖 | 多張 | log 域共用模型，支援 save/load |
| `denoise_PN2V.py` | 混合噪聲（Poisson + Gaussian） | 單張 | GMM 顯式建模噪聲分佈 |
| `denoise_PN2V_multi.py` | 混合噪聲，多圖 | 多張 | 共用 UNet + 共用 GMM |
| `denoise_apbsn.py` | 空間相關噪聲 / 掃描線 | 單張 | PD 域盲點訓練，無需 tiling |
| `denoise_apbsn_multi.py` | 空間相關噪聲，多圖 | 多張 | 同上，支援 save/load |
| `denoise_apbsn_faithful.py` | 真實複雜噪聲（論文完整版） | 單張 | DBSNl + L1 + 非對稱 PD + R3 |
| `denoise_apbsn_faithful_multi.py` | 真實複雜噪聲，多圖（論文完整版） | 多張 | 共用 DBSNl + 逐圖 R3，支援 save/load |
| `denoise_GR2R.py` | 未知加性噪聲 | 單張 | 再污染訓練對，完整感受野 |
| `denoise_GR2R_multi.py` | 未知加性噪聲，多圖 | 多張 | 共用模型，噪聲 σ 各圖估計後平均 |
| `denoise_DIP.py` | 無假設（任意） | 單張 | 無需噪聲模型，EMA 早停 |

---

## 快速對照表（N2V / PN2V 系列）

> **適用腳本：** `denoise_N2V.py`、`denoise_log_N2V.py`、`denoise_N2V_multi.py`、`denoise_PN2V.py`、`denoise_PN2V_multi.py`  
> AP-BSN、GR2R、DIP 請見各自章節。

| 影像大小 | `patch_size` | `batch_size` | `epochs` | `tile_size` | `tile_overlap` | 訓練時間（GPU） | 推論時間（GPU） |
|---|---|---|---|---|---|---|---|
| 64 × 64 | 32 | 32 | 300 | 64 | 0 | ~2 min | <1 s |
| 128 × 128 | 32 | 64 | 200 | 128 | 0 | ~2 min | <1 s |
| 256 × 256 | 64 | 64 | 150 | 256 | 0 | ~3 min | ~5 s |
| 512 × 512 | 64 | 128 | 100 | 256 | 48 | ~2 min | ~10 s |
| 1024 × 1024 | 64 | 128 | 100 | 256 | 48 | ~5 min | ~30 s |
| 2048 × 2048+ | 128 | 64 | 50 | 256 | 48 | ~15 min | ~2 min |
| CPU 執行 | 64 | 16 | 50 | 128 | 32 | ~1.5 hr | ~5 min |

---

## N2V 系列

### 參數選擇原則

#### patch_size

```
影像邊長 64   → patch_size 最大 32
影像邊長 128  → patch_size 最大 64（建議 32）
影像邊長 256+ → patch_size 64（標準）
影像邊長 2048+→ patch_size 128（可選）

規則：patch_size 必須為 8 的倍數，且 < 影像邊長的一半
```

#### tile_overlap

- 影像 ≤ 256px：`0`（整張一次處理）
- 影像 > 256px：`48`（約為 tile_size 的 20%）
- 有格狀拼接痕跡時增加至 `64–96`

---

### `denoise_N2V.py`（標準單張）

```bash
python denoise_N2V.py --input data/test_sem.tif
python denoise_N2V.py --input data/test_sem.tif --epochs 200 --patch_size 64 --batch_size 128
```

各影像大小建議設定請直接參照[快速對照表](#快速對照表n2v--pn2v-系列)。

---

### `denoise_log_N2V.py`（乘性 / 散斑噪聲）

**何時使用：** 影像亮區的顆粒感明顯強於暗區（噪聲隨信號強度變化），或已知存在散斑噪聲。

**原理：** 先對影像做 `log1p` 轉換，將乘性噪聲轉為加性 AWGN，再做標準 N2V 訓練；推論後以 `expm1` 還原。

```bash
python denoise_log_N2V.py --input data/test_sem.tif

# 參數與 denoise_N2V.py 完全相同（log 轉換在內部自動完成）
python denoise_log_N2V.py --input data/test_sem.tif --epochs 100 --patch_size 64 --batch_size 128
```

**注意事項：**
- 輸入影像中若有 0 值像素，`log1p` 會將其映射至 0，不影響結果
- 若影像同時包含乘性與加性噪聲，log 轉換可能欠補償加性成分 — 改用 `denoise_PN2V.py`

---

### `denoise_N2V_multi.py`（多張共用模型）

**何時使用：** 同一拍攝條件下有多張影像，希望共用一個噪聲模型以獲得更穩健的訓練。

```bash
# 基本用法：input_dir 的影像同時作為訓練集與推論集
python denoise_N2V_multi.py --input_dir ./sem_images --output_dir ./denoised

# 分開訓練集與推論集：先用高品質影像訓練，再對目標影像推論
python denoise_N2V_multi.py --train_dir ./reference_imgs --input_dir ./target_imgs --output_dir ./denoised

# 儲存 / 載入模型（跳過訓練直接推論）
python denoise_N2V_multi.py --input_dir ./sem_images --save_model model.pt
python denoise_N2V_multi.py --input_dir ./new_imgs --load_model model.pt --output_dir ./denoised
```

| 參數 | 預設值 | 說明 |
|---|---|---|
| `--input_dir` | `.` | 推論用影像目錄（未指定 `--train_dir` 時也作為訓練集） |
| `--train_dir` | （空） | 僅用於訓練的影像目錄；`--input_dir` 的影像仍全部做推論 |
| `--output_dir` | `denoised` | 輸出目錄 |
| `--save_model` | （空） | 訓練後儲存權重至此路徑（`.pt`） |
| `--load_model` | （空） | 載入預訓練權重並**跳過訓練** |

`patch_size`、`batch_size`、`epochs`、`tile_size`、`tile_overlap` 調整原則同單張版，參見[快速對照表](#快速對照表n2v--pn2v-系列)。

---

### `denoise_log_N2V_multi.py`（多張，乘性噪聲）

**何時使用：** 多張影像且噪聲為乘性 / 散斑型（亮區顆粒感明顯強於暗區）。  
共用一個 Log+N2V 模型，每張影像在 log 域獨立正規化後統一訓練。

```bash
python denoise_log_N2V_multi.py --input_dir ./sem_images --output_dir ./denoised

# 指定獨立訓練集
python denoise_log_N2V_multi.py --train_dir ./reference_imgs --input_dir ./target_imgs --output_dir ./denoised

# 儲存 / 載入
python denoise_log_N2V_multi.py --input_dir ./sem_images --save_model log_n2v_model.pt
python denoise_log_N2V_multi.py --input_dir ./new_imgs --load_model log_n2v_model.pt --output_dir ./denoised
```

| 參數 | 說明 |
|---|---|
| `--input_dir` | 推論用影像目錄（未指定 `--train_dir` 時同作訓練集） |
| `--train_dir` | 僅訓練用目錄（可與 `--input_dir` 不同） |
| `--output_dir` | 輸出目錄（預設 `denoised`） |
| `--save_model` / `--load_model` | 儲存 / 載入模型（`.pt`），`--load_model` 跳過訓練 |

**注意：** 若影像存在大量接近零的像素（暗背景比例 > 10%），腳本會自動印出 WARNING 並將像素值 floor 至 0.01 以穩定 log1p 轉換。  
`patch_size`、`batch_size`、`epochs` 等參數調整原則同 N2V，參見[快速對照表](#快速對照表n2v--pn2v-系列)。

---

## PN2V 系列

### 與 N2V 的差異

N2V 假設噪聲 MSE 可接受；**PN2V** 改以 **GMM（高斯混合模型）** 顯式建模噪聲的概率分佈，訓練目標改為最大化似然（NLL），能更準確處理 Poisson + Gaussian 混合噪聲。

> 訓練輸出的 loss 為 **NLL（負對數似然）**，數值與 N2V 的 MSE 不可比較。

---

### `denoise_PN2V.py`（單張，混合噪聲）

```bash
python denoise_PN2V.py --input data/test_sem.tif

# 高噪聲影像，增強 GMM 容量
python denoise_PN2V.py --input data/test_sem.tif --n_gaussians 5 --gmm_pretrain_epochs 500
```

#### PN2V 專屬核心參數

| 參數 | 預設值 | 說明 | 調整方向 |
|---|---|---|---|
| `--n_gaussians` | `3` | GMM 成分數 K（每個成分學習一種噪聲模式） | 見下方 K 選擇指南 |
| `--gmm_pretrain_epochs` | `300` | 僅用統計特性預訓練 GMM（不更新 UNet） | 若 GMM loss 在 300 epoch 後仍不穩定，增至 500 |
| `--infer_batch` | `8` | 推論時每批次處理的 tile 數（影響 VRAM） | OOM 時減至 4 或 2 |

#### `--n_gaussians`（K）選擇指南

| K 值 | 適用情境 |
|---|---|
| `2` | 已知兩種獨立噪聲源（如 Poisson 散粒 + Gaussian 讀出噪聲）；收斂最穩定 |
| `3`（預設） | 大多數真實 SEM 影像；第三成分吸收剩餘結構（輕微充電偽影、偵測器非線性） |
| `5` | 噪聲呈明顯多模態（強掃描線調製 + 散粒噪聲）；需更多 epochs，不建議用於 512×512 以下影像 |

**診斷 GMM 崩潰（K 太大）：** 若兩個成分的 σ(s) 曲線幾乎重疊 → 減小 K 並重新訓練。  
**診斷欠擬合（K 太小）：** 若 `val_NLL` 在 50 epochs 前就停止改善 → 增加 K。

`patch_size`、`batch_size`、`epochs`、`tile_size`、`tile_overlap` 調整原則同 N2V，參見[快速對照表](#快速對照表n2v--pn2v-系列)。

---

### `denoise_PN2V_multi.py`（多張，混合噪聲）

**何時使用：** 多張影像且懷疑存在混合噪聲；多圖像共用 UNet 與 GMM，噪聲統計更豐富。

```bash
python denoise_PN2V_multi.py --input_dir ./sem_images --output_dir ./denoised

# 指定獨立訓練集
python denoise_PN2V_multi.py --train_dir ./reference_imgs --input_dir ./target_imgs --output_dir ./denoised

# 儲存 / 載入
python denoise_PN2V_multi.py --input_dir ./sem_images --save_model pn2v_model.pt
python denoise_PN2V_multi.py --input_dir ./new_imgs --load_model pn2v_model.pt
```

| 參數 | 說明 |
|---|---|
| `--input_dir` | 推論用影像目錄（未指定 `--train_dir` 時同作訓練集） |
| `--train_dir` | 僅訓練用目錄（可與 `--input_dir` 不同） |
| `--output_dir` | 輸出目錄（預設 `denoised`） |
| `--n_gaussians` | GMM 成分數（預設 `3`，選擇方法同單張版） |
| `--gmm_pretrain_epochs` | GMM 預訓練輪數（預設 `300`） |
| `--save_model` / `--load_model` | 儲存 / 載入模型（`.pt`），`--load_model` 跳過訓練 |

---

## AP-BSN 系列

AP-BSN（CVPR 2022）先做 **PD（Pixel-Shuffle Downsampling）**：把原圖 `(H, W)` 拆成 r² 個空間交錯的子格，形成 `(r², H/r, W/r)` 的 PD 域影像。空間相關噪聲在 PD 域近似像素獨立，使盲點訓練得以在 PD 域運作。

> **注意：** AP-BSN 的 `patch_size` 是 **PD 域尺寸**（已縮小 r 倍），非原始像素大小。  
> 例如原圖 512×512、`pd_stride=2` → PD 域 256×256；此時 `patch_size=64` 對應原圖 128×128 的感受野。

---

### `denoise_apbsn.py`（單張）

```bash
# SEM 標準設定
python denoise_apbsn.py --input data/test_sem.tif

# 快速預覽（不需高品質）
python denoise_apbsn.py --avg_shifts False --epochs 50

# 相機 sRGB（空間相關 ISP 噪聲）
python denoise_apbsn.py --pd_stride 5 --batch_size 32 --avg_shifts True
```

#### AP-BSN 核心參數

| 參數 | 預設值 | 說明 | 調整方向 |
|---|---|---|---|
| `--pd_stride` | `2` | Pixel-Shuffle 步幅 r；決定 PD 域縮小比例與通道數（r²） | SEM → `2`；相機 sRGB 空間相關噪聲 → `5` |
| `--avg_shifts` | `True` | 是否平均 r² 種 shift 對齊（AP 品質模式） | `True`：消除 PD 格狀偽影；`False`：單次推論，快 r² 倍 |
| `--patch_size` | `64` | PD 域 patch 邊長（必須為 8 的倍數） | 太小上下文不足；太大吃 VRAM |
| `--batch_size` | `64` | 每次梯度更新的 patch 數 | r² 通道比 N2V 多，預設 64；OOM 時減半 |
| `--epochs` | `100` | 訓練 epoch 數 | 空間相關噪聲可增至 150 |

#### `pd_stride` 對影像維度的影響

| 原圖大小 | `pd_stride` | PD 域大小 | PD 通道數 | avg_shifts 推論次數 |
|---|---|---|---|---|
| 512 × 512 | 2 | 256 × 256 | 4 | 4 次 |
| 512 × 512 | 5 | ~102 × 102 | 25 | 25 次 |
| 1024 × 1024 | 2 | 512 × 512 | 4 | 4 次 |
| 2048 × 2048 | 2 | 1024 × 1024 | 4 | 4 次 |

> **無需 tiling**：PD 縮小讓整張影像可一次送入 GPU，不存在 tile 拼接問題。

#### AP-BSN 場景快速對照

| 使用場景 | `pd_stride` | `batch_size` | `epochs` | `avg_shifts` |
|---|---|---|---|---|
| SEM 像素獨立噪聲（標準） | `2` | `64` | `100` | `True` |
| 相機 sRGB 空間相關噪聲 | `5` | `32` | `100` | `True` |
| 快速預覽 | `2` | `64` | `50` | `False` |
| GPU RAM 不足（< 6 GB） | `2` | `32` | `100` | `True` |
| CPU 執行 | `2` | `16` | `50` | `False` |

#### AP-BSN 進階參數

| 參數 | 預設值 | 說明 |
|---|---|---|
| `mask_ratio` | `0.006` | 每個 patch 中被遮蔽的像素比例（約 0.6%）；一般不需調整 |
| `neighbor_radius` | `5` | 盲點替換時鄰居的最大位移；影像有明顯週期紋理時可增至 8–10 |

---

### `denoise_apbsn_multi.py`（多張）

**何時使用：** 多張影像且存在空間相關噪聲或掃描線問題。

```bash
python denoise_apbsn_multi.py --input_dir ./sem_images --output_dir ./denoised

# 快速模式（單次推論，省去 avg_shifts 的 r² passes）
python denoise_apbsn_multi.py --input_dir ./sem_images --fast

# 儲存 / 載入
python denoise_apbsn_multi.py --input_dir ./sem_images --save_model apbsn_model.pt
python denoise_apbsn_multi.py --input_dir ./new_imgs --load_model apbsn_model.pt
```

| 參數 | 說明 |
|---|---|
| `--input_dir` | 推論用影像目錄（未指定 `--train_dir` 時同作訓練集） |
| `--train_dir` | 僅訓練用目錄 |
| `--output_dir` | 輸出目錄（預設 `denoised`） |
| `--pd_stride` | PD 步幅（預設 `2`，選擇方法同單張版） |
| `--avg_shifts` | 預設開啟；與 `--fast` 互斥 |
| `--fast` | 開啟後強制單次推論（關閉 avg_shifts），速度快 r² 倍但可能留 PD 格紋 |
| `--save_model` / `--load_model` | 儲存 / 載入模型（`.pt`） |

---

### `denoise_apbsn_faithful.py`（單張，論文完整版）

**與 `denoise_apbsn.py` 的差異：**

| 項目 | `denoise_apbsn.py` | `denoise_apbsn_faithful.py` |
|---|---|---|
| 架構 | BSNUNet（標準 U-Net） | DBSNl（擴張盲點網路） |
| 盲點機制 | 訓練期遮罩（N2V 風格） | CentralMaskedConv2d（架構層級） |
| 損失函數 | MSE（僅遮罩像素） | L1（所有像素） |
| 推論後處理 | avg_shifts（平均位移） | R3 隨機替換細化（T=8, p=0.16） |
| PD 步幅 | 單一 pd_stride | pd_a（訓練）≠ pd_b（推論） |

```bash
# SEM 像素獨立噪聲（預設）
python denoise_apbsn_faithful.py --input data/test_sem.tif

# 相機 sRGB 空間相關噪聲
python denoise_apbsn_faithful.py --pd_a 5 --pd_b 2 --base_ch 128

# 快速預覽（跳過 R3）
python denoise_apbsn_faithful.py --no_r3
```

#### 核心參數

| 參數 | 預設值 | 說明 |
|---|---|---|
| `--pd_a` | `2` | 訓練 PD 步幅（SEM: 2；相機 sRGB: 5） |
| `--pd_b` | `2` | 推論 PD 步幅（論文建議: 2） |
| `--base_ch` | `64` | DBSNl 每分支通道數（論文: 128；SEM 用 64 已足夠） |
| `--num_module` | `9` | 每分支的 DCl 殘差塊數（論文: 9） |
| `--R3_T` | `8` | R3 細化次數（論文: 8） |
| `--R3_p` | `0.16` | R3 隨機替換比例（論文: 0.16） |
| `--no_r3` | — | 停用 R3，只做單次推論（快速預覽） |

#### 場景快速對照

| 使用場景 | `pd_a` | `pd_b` | `base_ch` | `epochs` | R3 |
|---|---|---|---|---|---|
| SEM 像素獨立噪聲 | `2` | `2` | `64` | `100` | 開啟 |
| 相機 sRGB 噪聲 | `5` | `2` | `128` | `100` | 開啟 |
| 快速預覽 | `2` | `2` | `32` | `50` | 關閉 |
| GPU RAM 不足（< 6 GB） | `2` | `2` | `32` | `100` | 開啟 |

---

### `denoise_apbsn_faithful_multi.py`（多張，論文完整版）

**何時使用：** 多張影像需要論文完整 DBSNl 架構（空間相關噪聲、相機 sRGB 噪聲、或對 `denoise_apbsn_multi.py` 效果不滿意時）。

```bash
python denoise_apbsn_faithful_multi.py \
    --input_dir ./sem_images --output_dir ./denoised

# 相機 sRGB 噪聲
python denoise_apbsn_faithful_multi.py \
    --input_dir ./sem_images --output_dir ./denoised \
    --pd_a 5 --pd_b 2 --base_ch 128

# 儲存 / 載入
python denoise_apbsn_faithful_multi.py \
    --input_dir ./sem_images --save_model apbsn_faithful.pt
python denoise_apbsn_faithful_multi.py \
    --input_dir ./new_imgs --load_model apbsn_faithful.pt
```

| 參數 | 說明 |
|---|---|
| `--input_dir` | 推論用影像目錄 |
| `--train_dir` | 僅訓練用目錄（未指定時同 `--input_dir`） |
| `--output_dir` | 輸出目錄（預設 `denoised`） |
| `--pd_a` / `--pd_b` | 訓練 / 推論 PD 步幅（同單張版） |
| `--base_ch` | DBSNl 通道數（預設 64） |
| `--no_r3` | 停用 R3（快速模式） |
| `--save_model` / `--load_model` | 儲存 / 載入模型（`.pt`）；`--load_model` 跳過訓練 |

---

## GR2R 參數設定

GR2R（CVPR 2021）不使用盲點遮蔽，改為對同一 patch 獨立做兩次「再污染」產生訓練對：

```
y1 = patch + α·σ·ε₁   (輸入)
y2 = patch + α·σ·ε₂   (目標，獨立採樣)
Loss = MSE(f(y1), y2)  — 覆蓋所有像素
```

> `patch_size`、`batch_size`、`epochs`、`tile_size`、`tile_overlap` 的調整原則與 N2V 完全相同，
> 請直接參照前面的[快速對照表](#快速對照表n2v--pn2v-系列)與[N2V 系列](#n2v-系列)。

### GR2R 專屬核心參數

| 參數 | 預設值 | 說明 | 調整方向 |
|---|---|---|---|
| `--alpha` | `1.0` | 再污染強度倍率：`σ_r = alpha × σ_estimated` | 見下方 alpha 選擇指南 |
| `--noise_std` | `0`（自動） | 手動指定噪聲 σ（0 = 用 Laplacian MAD 自動估計） | 有外部測量值時才手動指定 |
| `--poisson` | `False` | 改用 Poisson 再污染模式（適合散粒噪聲主導的 SEM） | 見下方噪聲模式選擇 |
| `--photon_scale` | `100` | Poisson 模式下的光子計數尺度（僅在 `--poisson` 時有效） | 見下方 photon_scale 估計 |

### `--alpha` 選擇指南

| alpha 值 | 效果 | 適用情境 |
|---|---|---|
| `0.5` | 輕度再污染；訓練信號較弱但偏差小 | 噪聲已很低；影像細節豐富需保留 |
| `1.0`（預設） | 再污染 σ 與原始噪聲相當；平衡點 | 大多數 SEM 影像的起點 |
| `1.5` | 較強再污染；訓練信號更穩定 | 高噪聲影像；loss 收斂不穩定時 |
| `2.0` | 強再污染；可能引入輕微模糊偏差 | 極高噪聲；噪聲估計不準確時的保守選擇 |

> **建議先以 `alpha=1.0` 執行，若輸出仍殘留可見噪聲再試 `1.5`；若輸出偏模糊則降至 `0.5`。**

### 噪聲模式選擇：Gaussian vs Poisson

| 情況 | 建議模式 | 原因 |
|---|---|---|
| 標準 SEM（混合噪聲） | Gaussian（預設） | MAD 估計的 σ 能涵蓋大部分加性噪聲 |
| 低電子劑量 SEM（高散粒噪聲） | `--poisson` | 散粒噪聲的方差與信號成正比，Poisson 再污染更符合物理 |
| 不確定時 | Gaussian | 先跑預設；若亮區殘留「顆粒感」強於暗區，再試 `--poisson` |

### `--photon_scale` 估計方法

```python
# 粗估方式：Poisson 噪聲滿足 Var(y) ≈ mean(y) / photon_scale
import numpy as np, tifffile
img = tifffile.imread('your_sem.tif').astype(float)
img = (img - img.min()) / (img.max() - img.min())
bright = img[img > 0.7]
photon_scale = bright.mean() / (bright.var() + 1e-10)
print(f"建議 photon_scale ≈ {photon_scale:.0f}")
```

| 估計值 | 對應 SEM 條件 |
|---|---|
| 50–100 | 低劑量、高速掃描 |
| 100–300（預設 100） | 一般 SEM 條件 |
| 300–1000 | 長時間積分、高劑量 |

> `photon_scale` 估計誤差在 2× 以內對結果影響有限，不需精確標定。

### GR2R 場景快速對照

| 使用場景 | `alpha` | `noise_std` | `--poisson` | `photon_scale` |
|---|---|---|---|---|
| 標準 SEM，未知噪聲 | `1.0` | `0`（自動） | 否 | — |
| 細節豐富，輕噪聲 | `0.5` | `0`（自動） | 否 | — |
| 高噪聲，收斂不穩 | `1.5` | `0`（自動） | 否 | — |
| 低劑量散粒噪聲主導 | `1.0` | — | 是 | `100`（或估計值） |
| 已知儀器噪聲 σ | `1.0` | 實測值 | 否 | — |

```bash
python denoise_GR2R.py --input data/test_sem.tif          # 標準（自動估計噪聲）
python denoise_GR2R.py --alpha 0.5                         # 輕噪聲，保留細節
python denoise_GR2R.py --poisson --photon_scale 80         # 高散粒噪聲
python denoise_GR2R.py --noise_std 0.04 --alpha 1.5        # 已知 σ，強化訓練
```

---

## `denoise_GR2R_multi.py`（多張）

**何時使用：** 多張影像，需要完整感受野訓練（不希望盲點遮蔽）；噪聲可為加性高斯或散粒噪聲。  
每張訓練影像各自估計噪聲 σ，取平均作為共用再污染強度。

```bash
python denoise_GR2R_multi.py --input_dir ./sem_images --output_dir ./denoised

# Poisson 再污染（散粒噪聲主導）
python denoise_GR2R_multi.py --input_dir ./sem_images --output_dir ./denoised --poisson

# 指定獨立訓練集
python denoise_GR2R_multi.py --train_dir ./reference_imgs --input_dir ./target_imgs --output_dir ./denoised

# 儲存 / 載入
python denoise_GR2R_multi.py --input_dir ./sem_images --save_model gr2r_model.pt
python denoise_GR2R_multi.py --input_dir ./new_imgs --load_model gr2r_model.pt --output_dir ./denoised
```

| 參數 | 說明 |
|---|---|
| `--input_dir` | 推論用影像目錄（未指定 `--train_dir` 時同作訓練集） |
| `--train_dir` | 僅訓練用目錄 |
| `--output_dir` | 輸出目錄（預設 `denoised`） |
| `--alpha` | 再污染強度倍率（預設 `1.0`，意義同單張版） |
| `--noise_std` | 手動噪聲 σ override（0 = 各圖估計後取平均） |
| `--poisson` | 改用 Poisson 再污染 |
| `--photon_scale` | Poisson 光子尺度（預設 `100`） |
| `--save_model` / `--load_model` | 儲存 / 載入模型（`.pt`） |

`alpha`、`noise_std`、`poisson`、`photon_scale` 的選擇方式與單張版完全相同，請參閱 [GR2R 參數設定](#gr2r-參數設定)。

---

## DIP 參數設定

Deep Image Prior（CVPR 2018）完全不使用噪聲模型假設：網路以隨機噪聲 z 作為固定輸入，反覆最佳化讓輸出擬合目標影像；網路的結構性歸納偏置會先學到低頻信號，在學到噪聲之前觸發 EMA 早停。

**無需 tiling**：DIP 每次前向傳播處理整張影像。對於超大影像（> 2048px）自動將 `num_channels` 降至 64 以控制 VRAM。

```bash
python denoise_DIP.py --input data/test_sem.tif

# 效果不足（噪聲殘留）→ 增加迭代數或放寬早停
python denoise_DIP.py --num_iterations 5000 --patience 100

# 效果過度（輸出模糊）→ 降低迭代數或收緊早停
python denoise_DIP.py --num_iterations 2000 --patience 30

# VRAM 不足（< 6 GB，影像 ≤ 2048px）→ 減少通道數
python denoise_DIP.py --num_channels 64
```

### DIP 核心參數

| 參數 | 預設值 | 說明 | 調整方向 |
|---|---|---|---|
| `--num_iterations` | `3000` | 最大梯度下降迭代次數 | 噪聲殘留 → 增至 5000；輸出模糊 → 降至 2000 |
| `--patience` | `50` | 連續 N 次迭代 loss > EMA 即觸發早停 | 噪聲殘留 → 增至 100；輸出模糊 → 降至 20 |
| `--min_iterations` | `500` | 早停前最少執行的迭代次數 | 一般不需調整 |
| `--num_channels` | `128` | UNet 編解碼器通道寬度 | 影像 > 2048px 自動降至 64；VRAM 不足時手動降 |
| `--num_levels` | `5` | 編碼器深度（stride-2 下採樣次數）；決定最小輸入倍數 = 2^num_levels | 一般不需調整；減少會降低感受野 |
| `--lr` | `0.01` | Adam 學習率（固定，不使用排程） | 收斂不穩 → 降至 `0.001`；收斂太慢 → 增至 `0.05` |
| `--reg_noise_std` | `0.03` | 每次迭代對 z 加入的擾動標準差（防止精確記憶） | 高噪聲影像 → 增至 `0.05`；低噪聲 → 降至 `0.01` |

### DIP 早停機制說明

```
EMA_loss(t) = 0.99 × EMA_loss(t−1) + 0.01 × loss(t)

若連續 patience 次滿足 loss(t) > EMA_loss(t)
且已執行 ≥ min_iterations 次
→ 觸發早停，取當前迭代的輸出
```

> DIP 通常在 1000–2000 次迭代間早停，總執行時間約 **3–5 分鐘**（GPU）。

### DIP 場景對照

| 使用場景 | `num_iterations` | `patience` | `num_channels` | `reg_noise_std` |
|---|---|---|---|---|
| 標準（預設） | `3000` | `50` | `128` | `0.03` |
| 高噪聲影像 | `5000` | `100` | `128` | `0.05` |
| 低噪聲，細節豐富 | `2000` | `30` | `128` | `0.01` |
| 影像 > 2048px | `3000` | `50` | `64`（自動） | `0.03` |
| VRAM < 6 GB | `3000` | `50` | `64` | `0.03` |

---

## 常見問題與調整

### GPU 顯示 OOM（VRAM 不足）

**N2V / PN2V / GR2R / Log-N2V（含多張版）：** 依序縮小以下參數：
1. `--tile_size`：256 → 128 → 64
2. `--batch_size`：減半
3. `--patch_size`：128 → 64

**PN2V 額外：** 降低 `--infer_batch`：8 → 4 → 2

**AP-BSN：** 無需 tiling；直接降低 `--batch_size`

**DIP：** 降低 `--num_channels`：128 → 64（影像 > 2048px 已自動執行）

---

### 訓練 loss 下降緩慢

- 增加 `--epochs`
- 確認影像已正規化至 [0, 1]（腳本內部會處理，但若自定義輸入需確認）
- N2V / GR2R：可提高 `patches_per_epoch`（`train_n2v()` 內變數）

---

### 輸出仍有明顯噪聲

| 腳本 | 可調整方向 |
|---|---|
| N2V 系列 | 增加 `--epochs`；擴大 `--patch_size` |
| PN2V | 增加 `--n_gaussians`（至 5）；增加 `--gmm_pretrain_epochs` |
| GR2R | 提高 `--alpha`（至 1.5）；或切換 `--poisson` |
| DIP | 增加 `--num_iterations`；增加 `--patience` |

---

### 輸出過度平滑（細節喪失）

| 腳本 | 可調整方向 |
|---|---|
| N2V 系列 | 減少 `--epochs` |
| GR2R | 降低 `--alpha`（至 0.5） |
| DIP | 降低 `--num_iterations`；降低 `--patience` 或 `--reg_noise_std` |

---

### 影像有明顯水平 / 垂直條紋（掃描線雜訊）

改用 `denoise_apbsn.py`（或 `denoise_apbsn_multi.py`），其 PD 域處理能有效抑制空間相關噪聲。  
若仍需用 N2V，可改用 `denoise_N2V_careamics.py` 並加入：

```python
struct_n2v_axis="horizontal"  # 或 "vertical"
```

---

### 有格狀拼接痕跡（tile 邊界可見）

1. 增加 `--tile_overlap`：48 → 64 → 96
2. 或縮小 `--tile_size` 使 tile 數增多，讓混合更充分

---

### PN2V GMM 不收斂

- 嘗試降低 `--n_gaussians`（5 → 3 → 2）
- 增加 `--gmm_pretrain_epochs`（300 → 500）
- 對較小影像（< 256px），建議 K ≤ 2

---

### DIP 早停觸發太早（迭代 < 500 次就停止）

`--min_iterations` 預設 500 次，不會提前停止。若實際輸出過於粗糙：
- 增加 `--patience`（50 → 100）
- 降低 `--lr`（0.01 → 0.001）使 loss 曲線更平滑

---

## 參數機制詳解

### 訓練參數（N2V / PN2V / GR2R / Log-N2V）

| 參數 | 預設值 | 機制 | 調整方向 |
|---|---|---|---|
| `--patch_size` | `64` | 從原圖隨機裁出 N×N 小塊作訓練單位；決定網路能看到的上下文範圍 | 太小（<32）上下文不足；太大（>128）吃 VRAM |
| `--batch_size` | `128` | 每次梯度更新同時處理的 patch 數；影響梯度穩定性與速度 | 大 batch → 穩定但吃 VRAM；OOM 時減半 |
| `--epochs` | `100` | 訓練資料集被完整跑過的次數 | 單張影像建議 100–200；太多可能過擬合噪聲 |
| `--tile_size` | `256` | 推論時切塊大小；越大邊界銜接越少但 VRAM 需求越高 | OOM 時：256 → 128 → 64 |
| `--tile_overlap` | `48` | 相鄰 tile 的重疊寬度（約 tile_size 的 20%）；防止邊界拼接痕跡 | 有格狀偽影時增加至 64–96 |

### 學習率設定

#### N2V / PN2V / GR2R：Adam + Cosine Annealing

```python
optimizer = optim.Adam(model.parameters(), lr=4e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
```

```
epoch   1 → lr ≈ 4e-4  （最大，快速學習）
epoch  50 → lr ≈ 2e-4  （中段）
epoch 100 → lr ≈ 1e-6  （最小，精細收斂）
```

- `4e-4` 是 N2V 社群驗證的經驗值，一般**不需調整**
- 若 loss 震盪不收斂 → 減小至 `1e-4`；收斂過慢 → 增至 `8e-4`

#### DIP：固定 Adam lr

DIP 使用固定 `lr=0.01`（無排程），依賴 EMA 早停而非 lr 衰減來控制訓練深度。  
調整 `--lr` 時，`0.001–0.05` 為合理範圍；超出此範圍可能影響早停行為。

### 驗證集分割（N2V 系列）

```
patches_per_epoch = 2000（固定於 train_n2v() 內）
    ├─ × 0.1 → val:   200 patch（只做前向傳播）
    └─ × 0.9 → train: 1800 patch
                      ÷ batch_size (128) → ~14 steps/epoch
                      × num_epochs (100) → ~1,400 次梯度更新
```

> N2V 是自監督，train/val 皆來自同一張圖；`val_loss` 主要作趨勢參考，不代表泛化能力。

---

## 解讀訓練輸出

### N2V / GR2R / Log-N2V（每 10 epoch 一行）

```
Epoch [  1/100]  train_loss=0.337753  val_loss=0.299617  elapsed=2.1s
Epoch [ 10/100]  train_loss=0.005203  val_loss=0.008156  elapsed=1.7s
Epoch [ 20/100]  train_loss=0.003526  val_loss=0.003599  elapsed=1.7s
```

#### `train_loss` / `val_loss`（MSE）

訓練集 / 驗證集上被遮蔽像素的 MSE（N2V 核心：遮住一個像素，用鄰居預測它）：

| `val_loss` vs `train_loss` | 代表 |
|---|---|
| `val ≈ train` | 正常收斂，泛化良好 |
| `val >> train` | 過擬合（網路記住訓練 patch 的噪聲） |
| `val` 持平但 `train` 仍下降 | 可考慮提早停止訓練 |

> **GR2R 的 loss 數值高於 N2V**：target `y2` 本身含噪聲，MSE 下限 = 再污染噪聲方差 `σ_r²`。只看趨勢，不需與 N2V 比較絕對值。

#### `elapsed`

單一 epoch 耗時（train + validate 合計）。第 1 epoch 較慢（CUDA kernel 初始化）。  
總時間估算：`elapsed × num_epochs`（如 1.7s × 100 ≈ 3 分鐘）

### PN2V（NLL loss）

```
Device: cuda  |  Parameters: 1,844,036
Pre-training GMM (3 Gaussians, 300 epochs)...
Epoch [  1/100]  train_NLL=-1.234  val_NLL=-1.198  elapsed=2.3s
Epoch [ 10/100]  train_NLL=-2.456  val_NLL=-2.401  elapsed=2.1s
```

PN2V 的 loss 為**負對數似然（NLL）**，數值越負代表擬合越好。與 N2V 的 MSE 不可直接比較。

### AP-BSN

```
Device: cuda  |  Parameters: 1,844,036
PD stride r=2: 4 channels  |  PD image: 256×256  (from 512×512)
patch_size=64  batch_size=64  epochs=100
Patches/epoch: train=1800  val=200
Epoch [  1/100]  train=0.001842  val=0.001956  1.8s
```

> AP-BSN loss 比 N2V 低約 1–2 個數量級（PD 域 r² 通道平均使 MSE 較小）；只看趨勢即可。

### DIP（每 100 次迭代一行）

```
Iter [  100/3000]  loss=0.004231  EMA=0.012456  elapsed=4.2s
Iter [  500/3000]  loss=0.002156  EMA=0.003201  elapsed=4.0s
Iter [  800/3000]  loss=0.002891  EMA=0.002734  [patience 1/50]
Early stopping at iter 850  (patience exhausted)
```

- `loss < EMA`：網路仍在學習信號（正常）
- `loss > EMA`：開始記憶噪聲，patience 計數器累加
- patience 耗盡後輸出當前結果

### 健康訓練的典型 loss 曲線（N2V 系列）

```
Epoch   1：loss 高（0.3+）  → 初始化，正常
Epoch  10：loss 大幅下降   → GPU 加速發揮，快速學習
Epoch  20：loss 趨穩，val ≈ train → 收斂健康
Epoch 100：loss 極低且穩定  → 訓練完成
```

若 `train_loss` 在 Epoch 50 後仍持續下降但 `val_loss` 不降，考慮減少 `--epochs` 或降低 learning rate。

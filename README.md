# SEM 影像去噪專案

基於 **Noise2Void (N2V)** 自監督深度學習的 SEM 影像去噪工具。不需要乾淨的參考影像，僅用單張帶噪影像即可訓練。

---

## 專案結構

```
Hu_denoise/
├── data/                      # 輸入影像與輸出結果（.tif、.png）
├── document/                  # 技術文件與說明指南（.md）
├── reference/                 # 學術論文 PDF 及引用索引
│
│   # ── 去噪腳本（單張影像）──
├── denoise_N2V.py             # 標準單張 N2V（PyTorch 基礎版）
├── denoise_PN2V.py            # PN2V 混合噪聲（GMM 噪聲模型）
├── denoise_log_N2V.py         # Log + N2V（Speckle 乘性噪聲）
├── denoise_apbsn.py           # AP-BSN（CVPR 2022，非對稱 PD）
├── denoise_DIP.py             # Deep Image Prior（CVPR 2018，無噪聲模型假設）
├── denoise_GR2R.py            # GR2R（CVPR 2021，雙重再破壞，全感受野）
│
│   # ── 去噪腳本（多張影像）──
├── denoise_N2V_multi.py       # 多張影像共用 N2V 模型
├── denoise_log_N2V_multi.py   # 多張影像 + 乘性噪聲，log 域共用模型
├── denoise_PN2V_multi.py      # 多張影像 + 混合噪聲，共用 GMM
├── denoise_GR2R_multi.py      # 多張影像 GR2R（全感受野，無盲點遮蔽）
├── denoise_apbsn_multi.py     # AP-BSN 多張影像版
├── denoise_apbsn_faithful.py  # AP-BSN 論文完整版（DBSNl + R3，單張）
├── denoise_apbsn_faithful_multi.py # AP-BSN 論文完整版（多張影像）
│
│   # ── 工具腳本 ──
├── test_sem.py                # 產生單張合成 SEM 測試影像
├── test_gen_multi.py          # 產生多張合成 SEM 測試影像
└── convert_to_tif.py          # PNG/JPG → TIFF 格式轉換
```

---

## 快速開始

```bash
# 安裝依賴
pip install torch tifffile matplotlib numpy
# 選用：pip install bm3d careamics

# 產生測試影像（若無真實 SEM 影像）
python test_sem.py

# 執行去噪（依噪聲類型選擇腳本，見下方說明）
python denoise_N2V.py
```

---

## 依噪聲類型選擇腳本

| 噪聲情境 | 推薦腳本 | 輸出 |
|---|---|---|
| 均勻顆粒噪聲（無條紋） | `denoise_N2V.py` | `data/denoised_sem_N2V.tif` |
| 混合噪聲（Poisson + Gamma）| `denoise_PN2V.py` ✦ | `data/denoised_sem_PN2V.tif` |
| 混合噪聲，自動選擇 GMM 容量 | `denoise_PN2V_bic.py` ✦ | `data/denoised_sem_PN2V_bic.tif` |
| Speckle / 乘性噪聲 | `denoise_log_N2V.py` | `data/denoised_sem_log_torch.tif` |
| 多張影像，相似條件 | `denoise_N2V_multi.py` | `--output_dir` 旗標指定 |
| 多張影像，乘性 / 散斑噪聲 | `denoise_log_N2V_multi.py` | `--output_dir` 旗標指定 |
| 多張影像，混合噪聲 | `denoise_PN2V_multi.py` ✦ | `--output_dir` 旗標指定 |
| 多張影像，混合噪聲，自動選擇 GMM 容量 | `denoise_PN2V_bic_multi.py` ✦ | `--output_dir` 旗標指定 |
| 多張影像，未知加性噪聲（全感受野） | `denoise_GR2R_multi.py` | `--output_dir` 旗標指定 |
| 真實世界複雜噪聲（非對稱 PD，抑制掃描格柵偽影） | `denoise_apbsn.py` | `data/denoised_apbsn.tif` |
| 真實世界複雜噪聲（論文完整版 DBSNl + R3） | `denoise_apbsn_faithful.py` | `data/denoised_sem_apbsn_faithful.tif` |
| 同上，多張影像 | `denoise_apbsn_faithful_multi.py` | `--output_dir` 旗標指定 |
| 真實世界複雜噪聲（無盲點遮罩，全感受野） | `denoise_GR2R.py` | `data/denoised_sem_GR2R.tif` |
| 噪聲類型未知 | `denoise_DIP.py` | `data/denoised_sem_DIP.tif` |
| N2V 留下棋盤格偽影 | `denoise_DIP.py` | `data/denoised_sem_DIP.tif` |

✦ 混合噪聲首選（GMM 直接對 Poisson-Gamma 原始分佈建模，無需 GAT 前處理）

**不確定噪聲類型時**：先用 BM3D 基準線快速評估視覺效果：
```python
import bm3d
result = bm3d.bm3d(image, sigma_psd=0.05)
```

---

## 腳本說明

### 單張影像

| 腳本 | 說明 |
|---|---|
| [denoise_N2V.py](denoise_N2V.py) | 純 PyTorch N2V 基礎版。適用一般均勻加性噪聲（高斯/泊松）。 |
| [denoise_PN2V.py](denoise_PN2V.py) | PN2V——GMM 噪聲模型，直接對 Poisson-Gamma 混合噪聲建模；含低計數診斷。與官方的主要差異：噪聲模型使用參數化 GMM（官方為非參數直方圖）；推論輸出單一純量（官方為 K=800 樣本後驗均值）。 |
| [denoise_PN2V_bic.py](denoise_PN2V_bic.py) | PN2V + BIC 自動選擇 GMM 容量——執行前自動評估 K ∈ {2,3,5,7} 的貝葉斯信息準則，選最低 BIC 的 K 再訓練。強 speckle（ENL < 3）時建議使用。 |
| [denoise_log_N2V.py](denoise_log_N2V.py) | 先用 `log1p` 將乘性 Speckle 轉為加性 AWGN，訓練後再用 `expm1` 還原。 |
| [denoise_apbsn.py](denoise_apbsn.py) | AP-BSN（CVPR 2022）——非對稱 PD + Blind-Spot Network；SEM 用 `pd_stride=2`，相機 sRGB 用 `pd_stride=5`。 |
| [denoise_apbsn_faithful.py](denoise_apbsn_faithful.py) | AP-BSN 論文完整版——DBSNl（CentralMaskedConv2d + 擴張分支）+ L1 全像素 loss + 非對稱 PD（pd_a 訓練 / pd_b 推論）+ R3 隨機替換細化（T=8, p=0.16）。 |
| [denoise_DIP.py](denoise_DIP.py) | Deep Image Prior（CVPR 2018）——無訓練集，以 generator 網路作隱式先驗，EMA 早停；GPU 約 3–5 分鐘。 |
| [denoise_GR2R.py](denoise_GR2R.py) | GR2R（CVPR 2021）——無盲點遮罩；訓練雙重再破壞 patch 對；全感受野；支援高斯與 Poisson 再破壞（`--poisson`）；自動估計噪聲標準差。 |

### 多張影像

| 腳本 | 說明 |
|---|---|
| [denoise_N2V_multi.py](denoise_N2V_multi.py) | 多張影像共用一個 N2V 模型（MSE loss），適用相似拍攝條件。 |
| [denoise_log_N2V_multi.py](denoise_log_N2V_multi.py) | 多張影像 Log+N2V，log 域共用訓練；適合乘性 / 散斑噪聲多圖場景。 |
| [denoise_PN2V_multi.py](denoise_PN2V_multi.py) | 多張影像共用 UNet + 共用 GMM；匯集所有影像的像素對，使噪聲統計更豐富。與官方差異同 denoise_PN2V.py（多張影像 shared GMM 為本專案擴展，非原論文內容）。 |
| [denoise_PN2V_bic_multi.py](denoise_PN2V_bic_multi.py) | 多張影像 PN2V + BIC 自動選擇 GMM 容量——BIC 在匯集所有影像的像素對上評估，統計支撐更充足。 |
| [denoise_GR2R_multi.py](denoise_GR2R_multi.py) | 多張影像 GR2R；無盲點遮蔽，全感受野；各圖自動估計 σ 後取均值作再污染強度。 |
| [denoise_apbsn_multi.py](denoise_apbsn_multi.py) | AP-BSN 多張影像版。 |
| [denoise_apbsn_faithful_multi.py](denoise_apbsn_faithful_multi.py) | AP-BSN 論文完整版多張影像——共用一個 DBSNl 模型訓練所有影像；R3 逐圖套用；支援 `--train_dir` / `--save_model` / `--load_model`。 |

```bash
# 多張影像去噪（N2V 或 PN2V 皆同介面）
python denoise_N2V_multi.py --input_dir ./sem_images --output_dir ./denoised
python denoise_PN2V_multi.py --input_dir ./sem_images --output_dir ./denoised
```

### 工具腳本

| 腳本 | 說明 |
|---|---|
| [test_sem.py](test_sem.py) | 產生 512×512 合成 SEM 測試影像（`data/test_sem.tif`），加入高斯噪聲。 |
| [test_gen_multi.py](test_gen_multi.py) | 產生多張合成 SEM 測試影像（用於多張影像測試）。 |
| [convert_to_tif.py](convert_to_tif.py) | 將 PNG/JPG/BMP/WebP 轉換為 TIFF，供去噪腳本使用。 |

```bash
# 轉換單一檔案（預設輸出灰階）
python convert_to_tif.py my_image.png --output data/

# 轉換整個資料夾
python convert_to_tif.py ./images --output data/

# 保留 RGB 色彩
python convert_to_tif.py my_image.jpg --keep-color --output data/
```

---

## 核心流程（N2V 系列）

1. **載入** — 讀取 `.tif`/`.tiff`/`.png`，RGB → 灰階，正規化為 `[0, 1]` float32
2. **分割** — 空間 80/20 切割為訓練/驗證區域（避免空間相關 patch 資料洩漏）
3. **訓練** — Blind-Spot N2V，向量化 numpy 遮罩（無 Python loop）
4. **預測** — 批次分塊推論，Hann-window 混合拼接，reflection padding 處理邊界與小影像
5. **輸出** — 儲存 `.tif` 並產生對比圖

---

## 參數調整參考

| 情境 | `patch_size` | `batch_size` | `num_epochs` |
|---|---|---|---|
| 單張影像，8GB GPU | `[64, 64]` | `64` | `200` |
| 多張影像，8GB GPU | `[64, 64]` | `128` | `100` |
| 高解析度（> 2048px） | `[128, 128]` | `32` | `100` |
| 僅 CPU | `[64, 64]` | `16` | `50` |

推論 OOM 時：將 `tile_size` 從 `[256,256]` → `[128,128]` → `[64,64]` 依序縮減。

---

## 環境需求

```bash
# 必要套件
pip install torch tifffile matplotlib numpy

# 選用套件
pip install scikit-learn  # denoise_PN2V_bic.py / denoise_PN2V_bic_multi.py 的 BIC 選擇
pip install bm3d          # BM3D 基準線評估
pip install careamics     # 舊版 CAREamics 腳本用

# NVIDIA GPU（CUDA 12.8）
pip install torch==2.9.1+cu128 torchvision==0.24.1+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
```

**Python 版本：** 3.12（建議）

---

## 技術文件

| 文件 | 說明 |
|---|---|
| [document/guide.md](document/guide.md) | N2V 技術背景、原理說明、Noise2Noise 系列方法介紹 |
| [document/N2V_optimization.md](document/N2V_optimization.md) | N2V 五項最佳化的詳細分析 |
| [document/speckle_denoising_strategy.md](document/speckle_denoising_strategy.md) | Speckle 乘性噪聲處理策略，含 Log 域轉換理論與實作 |
| [document/Speckle_vs_ShotNoise_Comparison.md](document/Speckle_vs_ShotNoise_Comparison.md) | Speckle 與 Shot Noise 去噪策略比較，含 SEM 實作指南 |
| [document/Mixed_Noise_Speckle_ShotNoise_Guide.md](document/Mixed_Noise_Speckle_ShotNoise_Guide.md) | Speckle + Shot Noise 混合去噪完整指南，含聯合處理策略 |
| [document/denoise_comparison.md](document/denoise_comparison.md) | 各腳本框架比較（演算法、速度、依賴套件、GPU 支援） |
| [document/settings_guide.md](document/settings_guide.md) | 參數調整指南（含 RTX 3080 時間估計） |
| [document/debug_report.md](document/debug_report.md) | 常見錯誤排除紀錄 |
| [document/critique.md](document/critique.md) | 噪聲處理策略的關鍵挑戰與模糊點分析 |

學術論文引用位於 [reference/](reference/) 資料夾（含 PDF 下載與索引）。

# SEM 影像去噪專案

基於 **Noise2Void (N2V)** 自監督深度學習的 SEM 影像去噪工具。不需要乾淨的參考影像，僅用單張帶噪影像即可訓練。

---

## 專案結構

```
Hu_denoise/
├── data/               # 輸入影像與輸出結果（.tif、.png）
├── document/           # 技術文件與說明指南（.md）
├── reference/          # 學術論文 PDF 及引用索引
├── denoise_N2V_careamics.py          # 主要去噪腳本（CAREamics，NVIDIA GPU）
├── denoise_N2V.py    # 純 PyTorch N2V（NVIDIA GPU）
├── denoise_apbsn.py    # AP-BSN CVPR 2022（空間相關噪聲）
├── denoise_log_N2V.py# Log + N2V（Speckle 噪聲）
├── denoise_N2V_intel_mkl.py  # CPU + Intel MKL（無 GPU）
├── denoise_N2V_tf.py       # TensorFlow/Keras N2V（僅 CPU）
├── test_sem.py         # 產生合成 SEM 測試影像至 data/
└── convert_to_tif.py   # PNG/JPG → TIFF 格式轉換
```

---

## 快速開始

```bash
# 啟動虛擬環境
h:/Hu_denoise/venv/Scripts/activate

# 產生測試影像（若無真實 SEM 影像，輸出至 data/test_sem.tif）
python test_sem.py

# 執行去噪（依 GPU 情況與噪聲類型選擇）
python denoise_N2V_careamics.py                    # NVIDIA GPU（CAREamics，一般噪聲）
python denoise_N2V.py              # NVIDIA GPU（純 PyTorch，一般噪聲）
python denoise_apbsn.py              # NVIDIA GPU（AP-BSN CVPR 2022，空間相關噪聲）
python denoise_log_N2V.py          # NVIDIA GPU（純 PyTorch，Speckle 噪聲）
python denoise_N2V_intel_mkl.py    # Intel CPU + MKL 最佳化（無 GPU 時）
```

所有腳本從 `data/` 讀取輸入影像，並將結果輸出到 `data/`：

| 腳本 | 輸入 | 輸出 TIF | 輸出 PNG |
|---|---|---|---|
| `denoise_N2V_careamics.py` | `data/test_sem.tif` | `data/denoised_sem.tif` | `data/denoising_result.png` |
| `denoise_N2V.py` | `data/test_sem.tif` | `data/denoised_sem_torch.tif` | `data/denoising_result.png` |
| `denoise_apbsn.py` | `data/test_sem.tif` | `data/denoised_sem_apbsn.tif` | `data/denoising_apbsn_result.png` |
| `denoise_log_N2V.py` | `data/test_sem.tif` | `data/denoised_sem_log_torch.tif` | `data/denoising_log_result.png` |
| `denoise_N2V_intel_mkl.py` | `data/test_sem.tif` | `data/denoised_sem_intel_mkl.tif` | `data/denoising_result_intel_mkl.png` |

若影像不是 TIFF 格式，先用 `convert_to_tif.py` 轉換，再放入 `data/`：

```bash
# 轉換單一檔案（預設輸出灰階）
python convert_to_tif.py my_image.png --output data/

# 轉換整個資料夾
python convert_to_tif.py ./images --output data/

# 保留 RGB 色彩
python convert_to_tif.py my_image.jpg --keep-color --output data/
```

---

## 腳本說明

### 主要執行腳本

| 檔案 | 框架 | GPU 支援 | 說明 |
|---|---|---|---|
| [denoise_N2V_careamics.py](denoise_N2V_careamics.py) | CAREamics | NVIDIA CUDA | 主要推薦版本，使用 CAREamics 高階框架執行 N2V |
| [denoise_N2V.py](denoise_N2V.py) | PyTorch | NVIDIA CUDA | 純 PyTorch N2V，適用一般加性噪聲（高斯/泊松） |
| [denoise_apbsn.py](denoise_apbsn.py) | PyTorch | NVIDIA CUDA | **AP-BSN**（CVPR 2022），Pixel-Shuffle Downsampling + Blind-Spot Network，適用空間相關噪聲；SEM 用 `pd_stride=2`，相機 sRGB 用 `pd_stride=5` |
| [denoise_log_N2V.py](denoise_log_N2V.py) | PyTorch | NVIDIA CUDA | **Log + N2V**，適用 Speckle 乘性噪聲（SEM 低能束/高倍率） |
| [denoise_N2V_intel_mkl.py](denoise_N2V_intel_mkl.py) | PyTorch + Intel MKL | CPU（Intel 最佳化） | 無 NVIDIA GPU 時的替代方案，透過 MKL-DNN / IPEX 最佳化 CPU 效能 |
| [denoise_N2V_tf.py](denoise_N2V_tf.py) | TensorFlow/Keras | 僅 CPU（Windows） | TF 版 N2V，Windows 不支援 GPU，需 WSL2 才能用 GPU |
| [test_sem.py](test_sem.py) | — | — | 產生 512×512 合成 SEM 測試影像（`data/test_sem.tif`），加入高斯噪聲 |
| [convert_to_tif.py](convert_to_tif.py) | — | — | 將 PNG/JPG/BMP/WebP 等格式轉換為 TIFF，供去噪腳本使用 |

### 測試腳本

| 檔案 | 說明 |
|---|---|
| [test_directml.py](test_directml.py) | `denoise_torch_directml.py` 的單元測試與 DirectML smoke test |

```bash
python test_directml.py -v          # 全部測試（不需 GPU）
python test_directml.py --benchmark # CPU vs DirectML 效能比較
```

### 文件說明

技術文件位於 [document/](document/) 資料夾：

| 檔案 | 說明 |
|---|---|
| [document/guide.md](document/guide.md) | N2V 技術背景、原理說明、Noise2Noise 系列方法介紹 |
| [document/denoise_comparison.md](document/denoise_comparison.md) | 去噪腳本的框架比較（演算法、速度、相依性、GPU 支援） |
| [document/settings_guide.md](document/settings_guide.md) | N2V 參數調整指南（patch_size、batch_size、epochs），含 RTX 3080 時間估計 |
| [document/speckle_denoising_strategy.md](document/speckle_denoising_strategy.md) | Speckle 乘性噪聲處理策略指南，基於 Noise2Noise 家族的理論分析與 Log 域轉換實作建議 |
| [document/Speckle_vs_ShotNoise_Comparison.md](document/Speckle_vs_ShotNoise_Comparison.md) | Speckle 與 Shot Noise 去噪策略比較，含 N2N 家族理論分析與 SEM 實作指南 |
| [document/Mixed_Noise_Speckle_ShotNoise_Guide.md](document/Mixed_Noise_Speckle_ShotNoise_Guide.md) | Speckle + Shot Noise 混合去噪完整指南，含聯合處理策略與運算資源比較 |
| [document/debug_report.md](document/debug_report.md) | 常見錯誤排除紀錄（模組未安裝、GPU 設定等） |

學術論文引用位於 [reference/](reference/) 資料夾（含 PDF 下載與索引）。

---

## 環境需求

```bash
# 共用套件
pip install careamics tifffile matplotlib numpy bm3d

# NVIDIA GPU（CUDA 12.8）
pip install torch==2.9.1+cu128 torchvision==0.24.1+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# Intel CPU 最佳化（MKL 內建於標準 PyTorch，IPEX 為選用）
pip install torch torchvision
pip install intel-extension-for-pytorch   # 選用，進一步提升 Intel CPU 效能

# TF 版（Windows 僅 CPU）
pip install tensorflow
```

**Python 版本：** 3.12（建議）

**GPU 對照表：**

| 顯卡類型 | 推薦腳本 | 備註 |
|---|---|---|
| NVIDIA RTX / GTX（一般噪聲） | `denoise_N2V_careamics.py` 或 `denoise_N2V.py` | CUDA 12.8 測試通過（RTX 3080） |
| NVIDIA RTX / GTX（空間相關噪聲） | `denoise_apbsn.py` | AP-BSN CVPR 2022；`pd_stride=2` SEM，`pd_stride=5` 相機 sRGB |
| NVIDIA RTX / GTX（Speckle） | `denoise_log_N2V.py` | Log + N2V，適用乘性 speckle |
| Intel Iris Xe / Arc（舊款 iGPU） | `denoise_N2V_intel_mkl.py` | CPU + MKL-DNN，選用 IPEX 進一步加速 |
| Intel UHD 620/630 | `denoise_N2V_intel_mkl.py` | iGPU 效益低，直接走 CPU 路徑更穩定 |
| 無獨顯 | 任一腳本 | 自動使用 CPU，速度較慢 |

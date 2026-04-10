# SEM 影像去噪專案

基於 **Noise2Void (N2V)** 自監督深度學習的 SEM 影像去噪工具。不需要乾淨的參考影像，僅用單張帶噪影像即可訓練。

---

## 快速開始

```bash
# 啟動虛擬環境
h:/Hu_denoise/venv/Scripts/activate

# 產生測試影像（若無真實 SEM 影像）
python test_sem.py

# 執行去噪（依 GPU 情況與噪聲類型選擇）
python denoise.py                    # NVIDIA GPU（CAREamics，一般噪聲）
python denoise_torch.py              # NVIDIA GPU（純 PyTorch，一般噪聲）
python denoise_apbsn.py              # NVIDIA GPU（AP-BSN CVPR 2022，空間相關噪聲）
python denoise_log_torch.py          # NVIDIA GPU（純 PyTorch，Speckle 噪聲）
python denoise_torch_intel_mkl.py    # Intel CPU + MKL 最佳化（無 GPU 時）
```

| 腳本 | 輸出 TIF | 輸出 PNG |
|---|---|---|
| `denoise.py` | `denoised_sem.tif` | `denoising_result.png` |
| `denoise_torch.py` | `denoised_sem_torch.tif` | `denoising_result.png` |
| `denoise_apbsn.py` | `denoised_sem_apbsn.tif` | `denoising_apbsn_result.png` |
| `denoise_log_torch.py` | `denoised_sem_log_torch.tif` | `denoising_log_result.png` |
| `denoise_torch_intel_mkl.py` | `denoised_sem_intel_mkl.tif` | `denoising_result_intel_mkl.png` |

若影像不是 TIFF 格式，先用 `convert_to_tif.py` 轉換：

```bash
# 轉換單一檔案（預設輸出灰階）
python convert_to_tif.py my_image.png

# 轉換整個資料夾
python convert_to_tif.py ./images --output ./tifs

# 保留 RGB 色彩
python convert_to_tif.py my_image.jpg --keep-color
```

---

## 腳本說明

### 主要執行腳本

| 檔案 | 框架 | GPU 支援 | 說明 |
|---|---|---|---|
| [denoise.py](denoise.py) | CAREamics | NVIDIA CUDA | 主要推薦版本，使用 CAREamics 高階框架執行 N2V |
| [denoise_torch.py](denoise_torch.py) | PyTorch | NVIDIA CUDA | 純 PyTorch N2V，適用一般加性噪聲（高斯/泊松） |
| [denoise_apbsn.py](denoise_apbsn.py) | PyTorch | NVIDIA CUDA | **AP-BSN**（CVPR 2022），Pixel-Shuffle Downsampling + Blind-Spot Network，適用空間相關噪聲；SEM 用 `pd_stride=2`，相機 sRGB 用 `pd_stride=5` |
| [denoise_log_torch.py](denoise_log_torch.py) | PyTorch | NVIDIA CUDA | **Log + N2V**，適用 Speckle 乘性噪聲（SEM 低能束/高倍率） |
| [denoise_torch_intel_mkl.py](denoise_torch_intel_mkl.py) | PyTorch + Intel MKL | CPU（Intel 最佳化） | 無 NVIDIA GPU 時的替代方案，透過 MKL-DNN / IPEX 最佳化 CPU 效能 |
| [denoise_tf.py](denoise_tf.py) | TensorFlow/Keras | 僅 CPU（Windows） | TF 版 N2V，Windows 不支援 GPU，需 WSL2 才能用 GPU |
| [test_sem.py](test_sem.py) | — | — | 產生 512×512 合成 SEM 測試影像（`test_sem.tif`），加入高斯噪聲 |
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

| 檔案 | 說明 |
|---|---|
| [guide.md](guide.md) | N2V 技術背景、原理說明、Noise2Noise 系列方法介紹 |
| [denoise_comparison.md](denoise_comparison.md) | 去噪腳本的框架比較（演算法、速度、相依性、GPU 支援） |
| [settings_guide.md](settings_guide.md) | N2V 參數調整指南（patch_size、batch_size、epochs），含 RTX 3080 時間估計 |
| [speckle_denoising_strategy.md](speckle_denoising_strategy.md) | Speckle 乘性噪聲處理策略指南，基於 Noise2Noise 家族的理論分析與 Log 域轉換實作建議 |
| [Speckle_vs_ShotNoise_Comparison.md](Speckle_vs_ShotNoise_Comparison.md) | Speckle 與 Shot Noise 去噪策略比較，含 N2N 家族理論分析與 SEM 實作指南 |
| [debug_report.md](debug_report.md) | 常見錯誤排除紀錄（模組未安裝、GPU 設定等） |
| [CLAUDE.md](CLAUDE.md) | 專案架構說明與 Claude Code 協作指引 |

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
| NVIDIA RTX / GTX（一般噪聲） | `denoise.py` 或 `denoise_torch.py` | CUDA 12.8 測試通過（RTX 3080） |
| NVIDIA RTX / GTX（空間相關噪聲） | `denoise_apbsn.py` | AP-BSN CVPR 2022；`pd_stride=2` SEM，`pd_stride=5` 相機 sRGB |
| NVIDIA RTX / GTX（Speckle） | `denoise_log_torch.py` | Log + N2V，適用乘性 speckle |
| Intel Iris Xe / Arc（舊款 iGPU） | `denoise_torch_intel_mkl.py` | CPU + MKL-DNN，選用 IPEX 進一步加速 |
| Intel UHD 620/630 | `denoise_torch_intel_mkl.py` | iGPU 效益低，直接走 CPU 路徑更穩定 |
| 無獨顯 | 任一腳本 | 自動使用 CPU，速度較慢 |

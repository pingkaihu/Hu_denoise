# SEM 影像去噪專案

基於 **Noise2Void (N2V)** 自監督深度學習的 SEM 影像去噪工具。不需要乾淨的參考影像，僅用單張帶噪影像即可訓練。

---

## 快速開始

```bash
# 啟動虛擬環境
h:/Hu_denoise/venv/Scripts/activate

# 產生測試影像（若無真實 SEM 影像）
python test_sem.py

# 執行去噪（依 GPU 情況選擇）
python denoise.py                    # NVIDIA GPU（CAREamics）
python denoise_torch.py              # NVIDIA GPU（純 PyTorch）
python denoise_torch_directml.py     # Intel / AMD iGPU（DirectML）
```

| 腳本 | 輸出 TIF | 輸出 PNG |
|---|---|---|
| `denoise.py` | `denoised_sem.tif` | `denoising_result.png` |
| `denoise_torch.py` | `denoised_sem_torch.tif` | `denoising_result.png` |
| `denoise_torch_directml.py` | `denoised_sem_directml.tif` | `denoising_result_directml.png` |

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
| [denoise_torch.py](denoise_torch.py) | PyTorch | NVIDIA CUDA | 純 PyTorch 手刻 N2V，無需 careamics |
| [denoise_torch_directml.py](denoise_torch_directml.py) | PyTorch + DirectML | Intel / AMD iGPU | 無 NVIDIA GPU 時的替代方案，透過 DirectML 使用內建顯示卡 |
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
| [denoise_comparison.md](denoise_comparison.md) | 三個去噪腳本的框架比較（速度、相依性、GPU 支援） |
| [settings_guide.md](settings_guide.md) | N2V 參數調整指南（patch_size、batch_size、epochs），含 RTX 3080 時間估計 |
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

# Intel / AMD iGPU（DirectML）
pip install torch torchvision torch-directml

# TF 版（Windows 僅 CPU）
pip install tensorflow
```

**Python 版本：** 3.12（建議）

**GPU 對照表：**

| 顯卡類型 | 推薦腳本 | 備註 |
|---|---|---|
| NVIDIA RTX / GTX | `denoise.py` 或 `denoise_torch.py` | CUDA 12.8 測試通過（RTX 3080） |
| Intel Iris Xe / Arc | `denoise_torch_directml.py` | 需安裝 `torch-directml` |
| Intel UHD 620/630（舊款） | `denoise_torch_directml.py`（CPU 模式） | iGPU 效益低，腳本會自動回退 CPU |
| 無獨顯 | 任一腳本 | 自動使用 CPU，速度較慢 |

# SEM 影像去噪專案

基於 **Noise2Void (N2V)** 自監督深度學習的 SEM 影像去噪工具。不需要乾淨的參考影像，僅用單張帶噪影像即可訓練。

---

## 快速開始

```bash
# 啟動虛擬環境
h:/Hu_denoise/venv/Scripts/activate

# 產生測試影像（若無真實 SEM 影像）
python test_sem.py

# 執行去噪
python denoise.py
```

輸出：`denoised_sem.tif`、`denoising_result.png`

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

| 檔案 | 框架 | 說明 |
|---|---|---|
| [denoise.py](denoise.py) | CAREamics | 主要推薦版本，使用 CAREamics 高階框架執行 N2V |
| [denoise_torch.py](denoise_torch.py) | PyTorch | 純 PyTorch 手刻 N2V，無需 careamics，Windows GPU 完整支援 |
| [denoise_tf.py](denoise_tf.py) | TensorFlow/Keras | TensorFlow 版 N2V，注意 Windows 原生不支援 GPU（需 WSL2） |
| [test_sem.py](test_sem.py) | — | 產生 512×512 合成 SEM 測試影像（`test_sem.tif`），加入高斯噪聲 |
| [convert_to_tif.py](convert_to_tif.py) | — | 將 PNG/JPG/BMP/WebP 等格式轉換為 TIFF，供去噪腳本使用 |

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
pip install careamics tifffile matplotlib numpy bm3d
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128  # CUDA GPU 版
pip install tensorflow  # TF 版（Windows 僅支援 CPU）
```

- Python 3.12
- NVIDIA GPU 建議（RTX 3080 10GB 測試通過）

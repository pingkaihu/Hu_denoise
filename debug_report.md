# denoise.py 除錯報告

**日期：** 2026-04-01  
**環境：** Windows 11, NVIDIA RTX 3080 (10 GB), Python venv

---

## 問題一：ModuleNotFoundError: No module named 'torch'

### 現象
```
ModuleNotFoundError: No module named 'torch'
```

### 原因
系統未安裝 PyTorch，虛擬環境缺少必要套件。

### 解決方式
在 venv 中安裝 PyTorch（指定 CUDA 版本）：
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 問題二：Python 3.14 不相容導致 scipy/torchmetrics 崩潰

### 現象
```
File "C:\Python314\Lib\multiprocessing\spawn.py" ...
scipy.linalg._decomp_lu_cython → KeyboardInterrupt / MemoryError
```

### 原因
虛擬環境使用 **Python 3.14**（過新），scipy、torchmetrics 等套件尚未提供 Python 3.14 的預編譯 wheel，Cython 擴充模組無法載入。

### 解決方式
刪除舊 venv，改用 **Python 3.12** 重建：
```powershell
py -3.12 -m venv venv
venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
venv\Scripts\pip install careamics tifffile matplotlib numpy bm3d
```

---

## 問題三：WinError 1455 — 分頁檔過小，CUDA DLL 載入失敗

### 現象
```
OSError: [WinError 1455] The paging file is too small for this operation to complete.
Error loading "...\torch\lib\cufft64_11.dll"
```

### 原因
`predict()` 會產生多個 multiprocessing worker 子程序。每個子程序重新 import `denoise.py` 時，module 層級的 `import torch` 觸發 CUDA DLL 載入（`cufft64_11.dll` 等），多個程序同時載入大型 DLL 導致 Windows 分頁檔耗盡。

### 解決方式
將 `import torch` 從 module 層級移入 `if __name__ == '__main__':` 區塊。worker 子程序 import 該檔案時不進入 `__main__`，因此不會載入 CUDA DLL。

```python
# 修改前（有問題）
import torch                        # ← 每個 worker 都會執行

if __name__ == '__main__':
    ...

# 修改後（正確）
if __name__ == '__main__':
    import torch                    # ← 只有主程序執行
    torch.set_float32_matmul_precision('high')
    ...
```

---

## 問題四：OpenBLAS 記憶體配置失敗

### 現象
```
OpenBLAS error: Memory allocation still failed after 10 retries, giving up.
```
發生於 sanity checking（訓練前驗證）階段。

### 原因
OpenBLAS 預設在每個 worker 程序中各自產生多個執行緒，多個 worker 同時配置 OpenBLAS 執行緒池，導致系統記憶體不足。

### 解決方式
在所有 import 之前設定環境變數，限制 OpenBLAS 與 OpenMP 執行緒數為 1：

```python
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
...
```

---

---

## 問題五：每個 Epoch 耗時數分鐘（DataLoader Worker 重啟開銷）

### 現象
訓練進度顯示每個 epoch 僅有 1 個 batch，但每個 epoch 實際耗時數分鐘。  
pytorch_lightning 警告：
```
Consider setting persistent_workers=True in 'val_dataloader'
Consider setting persistent_workers=True in 'train_dataloader'
```

### 原因
每個 epoch 結束後，DataLoader 的 worker 子程序被終止，下個 epoch 開始時重新 spawn。在 Windows 上，spawn 新 Python 程序的開銷極大（載入直譯器 + 所有套件），而實際 GPU 計算（1 batch, 333K UNet）不到 1 秒。因此訓練時間幾乎全部花在 worker 程序的重啟上。

### 解決方式
改用 `create_train_datamodule` 手動建立 DataModule，設定 `num_workers=0`，以單程序模式執行 DataLoader，完全避免 Windows 的程序 spawn 開銷：

```python
from careamics.lightning import create_train_datamodule

datamodule = create_train_datamodule(
    train_data=train_data,
    data_type="array",
    axes="SYX",
    patch_size=[64, 64],
    batch_size=128,
    val_percentage=0.1,
    train_dataloader_params={"num_workers": 0, "shuffle": True},
    val_dataloader_params={"num_workers": 0, "shuffle": False},
)

careamist.train(datamodule=datamodule)
```

---

---

## 問題六：ValueError: Mean and std must be provided in the configuration

### 現象
```
ValueError: Mean and std must be provided in the configuration.
```
發生於 `careamist.predict()` 呼叫時，訓練已正常完成（50 epochs）。

### 原因
使用 `careamist.train(train_source=...)` 時，CAREamics 會自動計算影像的 mean/std 並寫回 `cfg.data_config`。  
但改用自訂 `datamodule` 後（`careamist.train(datamodule=...)`），這個寫回步驟被繞過，config 內 `image_means` 與 `image_stds` 維持 `None`，`predict()` 讀取時報錯。

### 解決方式
訓練完成後，手動將 mean/std 設定回 config（值與訓練時印出的 `Computed dataset mean/std` 相同）：

```python
careamist.train(datamodule=datamodule)

careamist.cfg.data_config.set_means_and_stds(
    image_means=[float(train_data.mean())],
    image_stds=[float(train_data.std())],
)
```

---

## 修改摘要

| 檔案 | 修改內容 |
|---|---|
| `denoise.py` | 加入 `OPENBLAS_NUM_THREADS=1` / `OMP_NUM_THREADS=1`（所有 import 之前） |
| `denoise.py` | 將 `import torch`、GPU 檢查、`torch.set_float32_matmul_precision` 移入 `__main__` guard |
| `denoise.py` | 改用 `create_train_datamodule` + `num_workers=0`，避免 Windows worker spawn 開銷 |
| `denoise.py` | 訓練後手動呼叫 `set_means_and_stds()` 補回 predict 所需的正規化參數 |
| venv | 從 Python 3.14 改為 Python 3.12 重建 |

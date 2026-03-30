# ============================================================
# 安裝（只需執行一次）
# ============================================================
# !pip install careamics tifffile matplotlib numpy

# ============================================================
# 1. 載入與前處理 SEM 影像
# ============================================================
import torch
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from pathlib import Path

torch.set_float32_matmul_precision('high')  # 啟用 Tensor Core，提升 RTX 訓練速度

def load_sem_image(path: str) -> np.ndarray:
    """載入 SEM 影像，統一為 float32 灰階 numpy array"""
    img = tifffile.imread(path).astype(np.float32)
    
    # 若是 RGB 轉灰階
    if img.ndim == 3 and img.shape[-1] == 3:
        img = img @ np.array([0.2989, 0.5870, 0.1140])
    
    # 正規化到 [0, 1]
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img

# 讀取你的影像（支援 tif/tiff/png）
image_path = "test_sem.tif"
image = load_sem_image(image_path)
print(f"影像尺寸: {image.shape}, 範圍: [{image.min():.3f}, {image.max():.3f}]")

# ============================================================
# 2. 準備訓練資料（patch-based，從少量影像擷取大量 patches）
# ============================================================
from careamics import CAREamist
from careamics.config import create_n2v_configuration

# 建立 N2V 訓練設定
config = create_n2v_configuration(
    experiment_name="sem_n2v",
    data_type="array",           # 直接使用 numpy array
    axes="YX",                   # 2D 灰階
    patch_size=[64, 64],         # patch 大小（SEM 高解析度建議 64 或 128）
    batch_size=128,              # GPU 記憶體夠的話可調高
    num_epochs=100,              # 少量影像建議訓練久一點
)

# 初始化訓練器
careamist = CAREamist(source=config)

# ============================================================
# 3. 訓練（單張影像也可以，N2V 會自動切 patches）
# ============================================================
# 準備資料：shape 需為 (N, Y, X) 或 (Y, X)
train_data = image[np.newaxis, ...]  # 加 batch 維度 -> (1, H, W)

careamist.train(
    train_source=train_data,
    val_percentage=0.1,          # 10% 切為 validation
)

print("訓練完成！")

# ============================================================
# 4. 推論去噪
# ============================================================
denoised = careamist.predict(
    source=train_data,
    data_type="array",
    axes="YX",
    tile_size=[256, 256],        # 大圖分 tile 推論，避免 OOM
    tile_overlap=[48, 48],
)
denoised = np.squeeze(denoised)  # 移除多餘維度

# ============================================================
# 5. 視覺化比較
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image, cmap='gray')
axes[0].set_title('原始 SEM 影像')
axes[0].axis('off')

axes[1].imshow(denoised, cmap='gray')
axes[1].set_title('N2V 去噪結果')
axes[1].axis('off')

# 差異圖（放大 3x 方便觀察）
diff = np.abs(image - denoised) * 3
axes[2].imshow(diff, cmap='hot')
axes[2].set_title('差異圖 (×3)')
axes[2].axis('off')

plt.tight_layout()
plt.savefig("denoising_result.png", dpi=150)
plt.show()

# ============================================================
# 6. 儲存結果
# ============================================================
tifffile.imwrite("denoised_sem.tif", denoised)
print("已儲存至 denoised_sem.tif")

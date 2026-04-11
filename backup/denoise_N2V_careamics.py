# ============================================================
# 安裝（只需執行一次）
# ============================================================
# pip install careamics tifffile matplotlib numpy bm3d
# pip install torch==2.9.1+cu128 torchvision==0.24.1+cu128 --index-url https://download.pytorch.org/whl/cu128

# ============================================================
# 1. 載入與前處理 SEM 影像
# ============================================================
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import tifffile
import matplotlib.pyplot as plt
from pathlib import Path

def load_sem_image(path: str):
    """載入 SEM 影像，統一為 float32 灰階 numpy array，同時回傳原始 min/max 供還原用"""
    img = tifffile.imread(path).astype(np.float32)

    # 若是 RGB 轉灰階
    if img.ndim == 3 and img.shape[-1] == 3:
        img = img @ np.array([0.2989, 0.5870, 0.1140])

    # 保留原始值域，正規化到 [0, 1]
    img_min, img_max = float(img.min()), float(img.max())
    img = (img - img_min) / (img_max - img_min + 1e-8)
    return img, img_min, img_max


if __name__ == '__main__':
    # Windows multiprocessing 需要此 guard
    import torch
    torch.set_float32_matmul_precision('high')  # 啟用 Tensor Core，提升 RTX 訓練速度

    if torch.cuda.is_available():
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
        print(f"顯存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("警告：未偵測到 GPU，將使用 CPU 執行（速度較慢）")

    from careamics import CAREamist
    from careamics.config import create_n2v_configuration
    from careamics.lightning import create_train_datamodule

    # 讀取影像
    image_path = "data/test_sem.tif"
    image, img_min, img_max = load_sem_image(image_path)
    print(f"影像尺寸: {image.shape}, 原始範圍: [{img_min:.3f}, {img_max:.3f}]")

    # ============================================================
    # 2. 建立 N2V 訓練設定
    # ============================================================
    config = create_n2v_configuration(
        experiment_name="sem_n2v",
        data_type="array",
        axes="SYX",                  # S=sample, Y, X
        patch_size=[64, 64],
        batch_size=128,              # RTX 3080 10GB 足夠
        num_epochs=50,
    )

    careamist = CAREamist(source=config)

    # ============================================================
    # 3. 訓練
    # ============================================================
    train_data = image[np.newaxis, ...]  # (1, H, W)

    # num_workers=0：單程序 DataLoader，避免 Windows 每個 epoch 重新 spawn 子程序的開銷
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

    # 使用自訂 datamodule 訓練時，mean/std 不會自動寫回 config，需手動設定
    careamist.cfg.data_config.set_means_and_stds(
        image_means=[float(train_data.mean())],
        image_stds=[float(train_data.std())],
    )
    print("訓練完成！")

    # ============================================================
    # 4. 推論去噪
    # ============================================================
    denoised = careamist.predict(
        source=train_data,
        data_type="array",
        axes="SYX",
        tile_size=[256, 256],
        tile_overlap=[48, 48],
    )
    denoised = np.squeeze(denoised)

    # 還原至原始灰階值域
    denoised_original = (denoised * (img_max - img_min) + img_min).astype(np.float32)

    # ============================================================
    # 5. 儲存結果
    # ============================================================
    tifffile.imwrite("data/denoised_sem.tif", denoised_original)
    print(f"已儲存至 data/denoised_sem.tif，輸出範圍: [{denoised_original.min():.3f}, {denoised_original.max():.3f}]")


    # ============================================================
    # 6. 視覺化比較
    # ============================================================
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Originl SEM')
    ax[0].axis('off')

    ax[1].imshow(denoised, cmap='gray')
    ax[1].set_title('N2V Denoised')
    ax[1].axis('off')

    diff = np.abs(image - denoised) * 3
    ax[2].imshow(diff, cmap='hot')
    ax[2].set_title('Difference (mulitplied by 3)')
    ax[2].axis('off')

    plt.tight_layout()
    plt.savefig("data/denoising_result.png", dpi=150)
    plt.show()


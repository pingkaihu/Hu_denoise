import numpy as np
import tifffile

# 模擬 SEM 影像（高斯噪聲加在結構影像上）
rng = np.random.default_rng(42)
structure = np.zeros((512, 512), dtype=np.float32)
for _ in range(20):
    x, y = rng.integers(50, 462, size=2)
    r = rng.integers(20, 60)
    yy, xx = np.ogrid[:512, :512]
    structure[((xx - x)**2 + (yy - y)**2) < r**2] = 1.0

noisy = structure + rng.normal(0, 0.1, structure.shape).astype(np.float32)
import os
os.makedirs("data", exist_ok=True)
tifffile.imwrite("data/test_sem.tif", noisy)
print("已產生 data/test_sem.tif")

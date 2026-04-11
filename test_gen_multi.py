# ============================================================
# Generate 10 synthetic SEM-like noisy images for testing
# Output: data/test_multi/sem_00.tif ... sem_09.tif
# ============================================================

import numpy as np
import tifffile
from pathlib import Path

OUT_DIR = Path("data/test_multi")
OUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(0)

for i in range(10):
    structure = np.zeros((512, 512), dtype=np.float32)

    # Random circles (different per image → different "objects")
    n_circles = rng.integers(10, 30)
    for _ in range(n_circles):
        x, y = rng.integers(50, 462, size=2)
        r    = rng.integers(15, 70)
        yy, xx = np.ogrid[:512, :512]
        structure[((xx - x)**2 + (yy - y)**2) < r**2] = rng.uniform(0.5, 1.0)

    # Add Gaussian noise (same sigma across all images → same noise statistics)
    noisy = structure + rng.normal(0, 0.1, structure.shape).astype(np.float32)
    noisy = np.clip(noisy, 0, None).astype(np.float32)

    path = OUT_DIR / f"sem_{i:02d}.tif"
    tifffile.imwrite(str(path), noisy)
    print(f"Generated: {path}")

print(f"\nDone. 10 images saved to '{OUT_DIR}/'")

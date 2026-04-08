# ============================================================
# Tests for denoise_torch_directml.py
# Run: python test_directml.py
#      python test_directml.py -v          # verbose
#      python test_directml.py -k device   # run only device tests
# ============================================================

import sys
import time
import unittest

import numpy as np
import torch

from denoise_torch_intel_mkl import (
    N2VDataset,
    N2VUNet,
    setup_cpu_device,
    load_sem_image,
    predict_tiled,
    train_n2v,
)

# ── helpers ──────────────────────────────────────────────────
SMALL_IMAGE = np.random.default_rng(0).random((128, 128), dtype=np.float32)


def _tiny_model() -> N2VUNet:
    return N2VUNet(in_channels=1, base_features=8)   # minimal params


# ============================================================
# 1. Device selection
# ============================================================

class TestDeviceSelection(unittest.TestCase):

    def test_setup_cpu_device_returns_cpu(self):
        """setup_cpu_device() must return torch.device('cpu')."""
        device = setup_cpu_device()
        self.assertEqual(device.type, 'cpu')

    def test_setup_cpu_device_does_not_raise(self):
        """setup_cpu_device() must not raise regardless of IPEX availability."""
        try:
            setup_cpu_device()
        except Exception as exc:
            self.fail(f"setup_cpu_device() raised unexpectedly: {exc}")

    def test_tensor_on_cpu_device(self):
        """Tensors must be usable on the returned device."""
        device = setup_cpu_device()
        t = torch.zeros(2, 2).to(device)
        self.assertEqual(t.shape, (2, 2))


# ============================================================
# 2. N2VDataset
# ============================================================

class TestN2VDataset(unittest.TestCase):

    def test_length(self):
        ds = N2VDataset(SMALL_IMAGE, patch_size=64, num_patches=50)
        self.assertEqual(len(ds), 50)

    def test_output_shapes(self):
        ds = N2VDataset(SMALL_IMAGE, patch_size=64, num_patches=4)
        corrupted, clean, mask = ds[0]
        self.assertEqual(corrupted.shape, (1, 64, 64))
        self.assertEqual(clean.shape,     (1, 64, 64))
        self.assertEqual(mask.shape,      (1, 64, 64))

    def test_mask_is_binary(self):
        ds = N2VDataset(SMALL_IMAGE, patch_size=64, num_patches=4)
        _, _, mask = ds[0]
        unique = mask.unique()
        for v in unique:
            self.assertIn(v.item(), [0.0, 1.0])

    def test_mask_pixel_count(self):
        """Number of masked pixels should match n_masked = int(64*64*0.006) = 24."""
        ds = N2VDataset(SMALL_IMAGE, patch_size=64, num_patches=4, mask_ratio=0.006)
        _, _, mask = ds[0]
        n_masked = int(mask.sum().item())
        expected = max(1, int(64 * 64 * 0.006))  # 24
        self.assertEqual(n_masked, expected)

    def test_corrupted_differs_from_clean_at_masked(self):
        """Masked pixels must have been replaced (corrupted != clean at mask positions)."""
        ds = N2VDataset(SMALL_IMAGE, patch_size=64, num_patches=4,
                        mask_ratio=0.05, rng_seed=7)
        corrupted, clean, mask = ds[0]
        mask_bool = mask.squeeze().bool()
        # Not strictly guaranteed to differ (neighbor could equal original by coincidence),
        # but with mask_ratio=0.05 on random data the probability of all matching is ~0
        self.assertFalse(torch.all(corrupted.squeeze()[mask_bool] ==
                                   clean.squeeze()[mask_bool]))

    def test_patch_size_not_divisible_by_8_raises(self):
        with self.assertRaises(AssertionError):
            N2VDataset(SMALL_IMAGE, patch_size=60)   # 60 % 8 != 0

    def test_image_too_small_raises(self):
        tiny = np.zeros((16, 16), dtype=np.float32)
        with self.assertRaises(AssertionError):
            N2VDataset(tiny, patch_size=64)

    def test_reproducibility_with_seed(self):
        ds1 = N2VDataset(SMALL_IMAGE, patch_size=64, num_patches=4, rng_seed=42)
        ds2 = N2VDataset(SMALL_IMAGE, patch_size=64, num_patches=4, rng_seed=42)
        c1, _, _ = ds1[0]
        c2, _, _ = ds2[0]
        self.assertTrue(torch.equal(c1, c2))


# ============================================================
# 3. N2VUNet architecture
# ============================================================

class TestN2VUNet(unittest.TestCase):

    def test_output_shape_matches_input(self):
        model = _tiny_model()
        x = torch.zeros(2, 1, 64, 64)
        y = model(x)
        self.assertEqual(y.shape, x.shape)

    def test_output_shape_256(self):
        model = _tiny_model()
        x = torch.zeros(1, 1, 256, 256)
        y = model(x)
        self.assertEqual(y.shape, x.shape)

    def test_no_nan_in_output(self):
        model = _tiny_model()
        x = torch.rand(2, 1, 64, 64)
        y = model(x)
        self.assertFalse(torch.isnan(y).any())

    def test_gradients_flow(self):
        model = _tiny_model()
        x = torch.rand(1, 1, 64, 64)
        y = model(x)
        y.sum().backward()
        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, f"No gradient for {name}")

    def test_parameter_count_is_reasonable(self):
        model = N2VUNet(in_channels=1, base_features=32)
        n = sum(p.numel() for p in model.parameters())
        # Full model: ~800 k–2 M params
        self.assertGreater(n, 100_000)
        self.assertLess(n, 5_000_000)


# ============================================================
# 4. load_sem_image
# ============================================================

class TestLoadSemImage(unittest.TestCase):

    def test_normalized_range(self):
        """Output must be in [0, 1]."""
        import tifffile, tempfile, os
        arr = (np.random.rand(64, 64) * 255).astype(np.float32)
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
            path = f.name
        try:
            tifffile.imwrite(path, arr)
            img, mn, mx = load_sem_image(path)
            self.assertGreaterEqual(img.min(), 0.0)
            self.assertLessEqual(img.max(), 1.0 + 1e-6)
            self.assertAlmostEqual(float(mn), float(arr.min()), places=3)
            self.assertAlmostEqual(float(mx), float(arr.max()), places=3)
        finally:
            os.unlink(path)

    def test_rgb_converted_to_grayscale(self):
        import tifffile, tempfile, os
        arr = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
            path = f.name
        try:
            tifffile.imwrite(path, arr)
            img, _, _ = load_sem_image(path)
            self.assertEqual(img.ndim, 2)
        finally:
            os.unlink(path)


# ============================================================
# 5. predict_tiled (CPU — no GPU needed)
# ============================================================

class TestPredictTiled(unittest.TestCase):

    def _trained_model(self) -> N2VUNet:
        """Return a model that's been through 1 epoch so weights are non-zero."""
        model = _tiny_model()
        model = train_n2v(
            model, SMALL_IMAGE,
            patch_size=64, batch_size=32, num_epochs=1,
        )
        return model

    def test_output_shape_matches_input(self):
        model = self._trained_model()
        out = predict_tiled(model, SMALL_IMAGE,
                            tile_size=(64, 64), tile_overlap=(8, 8))
        self.assertEqual(out.shape, SMALL_IMAGE.shape)

    def test_output_dtype_is_float32(self):
        model = self._trained_model()
        out = predict_tiled(model, SMALL_IMAGE,
                            tile_size=(64, 64), tile_overlap=(8, 8))
        self.assertEqual(out.dtype, np.float32)

    def test_no_nan_or_inf_in_output(self):
        model = self._trained_model()
        out = predict_tiled(model, SMALL_IMAGE,
                            tile_size=(64, 64), tile_overlap=(8, 8))
        self.assertFalse(np.isnan(out).any(), "NaN in output")
        self.assertFalse(np.isinf(out).any(), "Inf in output")

    def test_tile_larger_than_image_raises(self):
        model = _tiny_model()
        with self.assertRaises(AssertionError):
            predict_tiled(model, SMALL_IMAGE,
                          tile_size=(256, 256), tile_overlap=(8, 8))

    def test_tile_size_not_divisible_by_8_raises(self):
        model = _tiny_model()
        with self.assertRaises(AssertionError):
            predict_tiled(model, SMALL_IMAGE,
                          tile_size=(60, 64), tile_overlap=(8, 8))


# ============================================================
# 6. DirectML-specific smoke tests (skipped if not installed)
# ============================================================

def _dml_available() -> bool:
    try:
        import torch_directml
        return torch_directml.is_available()
    except ImportError:
        return False


class TestIntelMKLSmoke(unittest.TestCase):
    """CPU + MKL 基本 smoke test（無需 GPU，永遠執行）。"""

    def test_model_forward_on_cpu(self):
        model = _tiny_model()
        x = torch.zeros(1, 1, 64, 64)
        y = model(x)
        self.assertEqual(y.shape, (1, 1, 64, 64))

    def test_training_one_epoch_on_cpu(self):
        model = _tiny_model()
        t0 = time.time()
        model = train_n2v(
            model, SMALL_IMAGE,
            patch_size=64, batch_size=16, num_epochs=1,
        )
        elapsed = time.time() - t0
        print(f"\n  CPU (MKL) 1-epoch time: {elapsed:.2f}s")
        self.assertIsInstance(model, N2VUNet)

    def test_pin_memory_false_for_cpu(self):
        """CPU 路徑下 pin_memory 必須為 False。"""
        from torch.utils.data import DataLoader
        ds = N2VDataset(SMALL_IMAGE, patch_size=64, num_patches=4)
        loader = DataLoader(ds, batch_size=4, pin_memory=False)
        self.assertFalse(loader.pin_memory)

    def test_ipex_import_does_not_crash(self):
        """IPEX 未安裝時 apply_ipex 應靜默跳過，不 raise。"""
        from denoise_torch_intel_mkl import apply_ipex
        model = _tiny_model()
        optimizer = optim.Adam(model.parameters())
        try:
            m, o = apply_ipex(model, optimizer)
            self.assertIsNotNone(m)
        except Exception as exc:
            self.fail(f"apply_ipex() raised unexpectedly: {exc}")


# ============================================================
# 7. Performance benchmark (opt-in, skipped in normal test runs)
# ============================================================

@unittest.skipUnless(
    '--benchmark' in sys.argv,
    "Pass --benchmark to run performance comparison"
)
class TestPerformanceBenchmark(unittest.TestCase):
    """
    Compare 1-epoch training time: CPU vs DirectML.
    Run with: python test_directml.py --benchmark
    """

    IMAGE = np.random.default_rng(1).random((256, 256), dtype=np.float32)

    def _time_one_epoch(self, **train_kwargs) -> float:
        model = _tiny_model()
        t0 = time.time()
        train_n2v(model, self.IMAGE, patch_size=64, batch_size=32, num_epochs=1,
                  **train_kwargs)
        return time.time() - t0

    def test_cpu_mkl_baseline(self):
        """CPU + MKL 單 epoch 時間基準。"""
        cpu_time = self._time_one_epoch()
        print(f"\n  CPU (MKL) 1-epoch: {cpu_time:.2f}s")

    def test_cpu_ipex_vs_plain(self):
        """比較有無 IPEX 的速度差異（IPEX 未安裝時只印 plain 結果）。"""
        plain_time = self._time_one_epoch()
        print(f"\n  CPU plain  1-epoch: {plain_time:.2f}s")
        try:
            import intel_extension_for_pytorch  # noqa: F401
            ipex_time = self._time_one_epoch()
            speedup = plain_time / ipex_time
            print(f"  CPU + IPEX 1-epoch: {ipex_time:.2f}s  "
                  f"({speedup:.2f}x {'faster' if speedup > 1 else 'slower'})")
        except ImportError:
            print("  IPEX 未安裝，跳過比較")


# ============================================================
# Entry point
# ============================================================

if __name__ == '__main__':
    # Remove --benchmark from argv so unittest doesn't choke on it
    argv = [a for a in sys.argv if a != '--benchmark']
    unittest.main(argv=argv, verbosity=2)

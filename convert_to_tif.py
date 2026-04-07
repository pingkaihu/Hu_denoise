"""
convert_to_tif.py — Convert common image formats to TIFF

Usage:
    python convert_to_tif.py <input>            # single file or directory
    python convert_to_tif.py <input> --output <dir>
    python convert_to_tif.py <input> --keep-color

Supported input formats: PNG, JPG/JPEG, BMP, GIF, WebP
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}


def to_grayscale(arr: np.ndarray) -> np.ndarray:
    """Convert an array to grayscale, preserving the original dtype."""
    if arr.ndim == 2:
        return arr  # already single-channel
    # Drop alpha, then weighted sum
    rgb = arr[:, :, :3]
    if np.issubdtype(arr.dtype, np.floating):
        return (0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]).astype(arr.dtype)
    # Integer types: compute in float64 then round back
    gray = 0.2989 * rgb[:, :, 0].astype(np.float64) + \
           0.5870 * rgb[:, :, 1].astype(np.float64) + \
           0.1140 * rgb[:, :, 2].astype(np.float64)
    return np.round(gray).astype(arr.dtype)


def normalize_array(arr: np.ndarray) -> np.ndarray:
    """
    Normalize image data to a TIFF-friendly representation:
    - float:  min-max scale to [0.0, 1.0] float32
    - int32 (from 16-bit PNG via PIL mode I): cast to uint16
    - uint8 / uint16: keep as-is
    """
    if np.issubdtype(arr.dtype, np.floating):
        lo, hi = arr.min(), arr.max()
        if hi > lo:
            arr = (arr - lo) / (hi - lo)
        else:
            arr = np.zeros_like(arr)
        return arr.astype(np.float32)
    if arr.dtype == np.int32:
        # PIL mode "I" wraps 16-bit PNGs as int32; clip to uint16 range
        return arr.clip(0, 65535).astype(np.uint16)
    return arr  # uint8 or uint16 — already fine


def convert_file(src: Path, out_dir: Path, keep_color: bool) -> bool:
    try:
        img = Image.open(src)
        # GIF: use first frame only
        if hasattr(img, "n_frames") and img.n_frames > 1:
            img.seek(0)

        arr = np.array(img)  # preserves original dtype (float32, int32, uint8, …)

        if not keep_color:
            arr = to_grayscale(arr)

        arr = normalize_array(arr)

        dst = out_dir / (src.stem + ".tif")
        tifffile.imwrite(str(dst), arr)
        print(f"  OK  {src.name}  [{img.mode} {arr.dtype}]  ->  {dst}")
        return True
    except Exception as e:
        print(f"  FAIL  {src.name}: {e}", file=sys.stderr)
        return False


def collect_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(f"Unsupported format: {input_path.suffix}", file=sys.stderr)
            sys.exit(1)
        return [input_path]
    elif input_path.is_dir():
        files = [
            p for p in sorted(input_path.iterdir())
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if not files:
            print(f"No supported images found in {input_path}", file=sys.stderr)
            sys.exit(1)
        return files
    else:
        print(f"Path not found: {input_path}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Convert images to TIFF format")
    parser.add_argument("input", help="Image file or directory")
    parser.add_argument("--output", "-o", help="Output directory (default: same as source)")
    parser.add_argument("--keep-color", action="store_true",
                        help="Keep RGB channels (default: convert to grayscale)")
    args = parser.parse_args()

    input_path = Path(args.input)
    files = collect_images(input_path)

    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        # For a single file, output beside it; for a directory, output inside it
        out_dir = files[0].parent if input_path.is_file() else input_path

    print(f"Converting {len(files)} file(s) -> {'RGB' if args.keep_color else 'grayscale'} TIFF")
    ok = sum(convert_file(f, out_dir, args.keep_color) for f in files)
    failed = len(files) - ok
    print(f"\nDone: {ok} converted, {failed} failed.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

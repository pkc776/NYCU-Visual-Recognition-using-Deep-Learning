from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="pred.npz")
    parser.add_argument("--test-dir", default="release_folder/hw4_realse_dataset/test/degraded")
    args = parser.parse_args()

    data = np.load(args.npz)
    test_files = sorted(Path(args.test_dir).glob("*.png"), key=lambda p: int(p.stem))
    expected = [p.name for p in test_files]
    keys = sorted(data.files, key=lambda x: int(Path(x).stem))
    if keys != expected:
        raise ValueError(f"Key mismatch. expected first={expected[:5]} got first={keys[:5]}")
    for path in test_files:
        arr = data[path.name]
        with Image.open(path) as image:
            w, h = image.size
        if arr.shape != (3, h, w):
            raise ValueError(f"{path.name}: shape {arr.shape}, expected {(3, h, w)}")
        if arr.dtype != np.uint8:
            raise ValueError(f"{path.name}: dtype {arr.dtype}, expected uint8")
        if arr.min() < 0 or arr.max() > 255:
            raise ValueError(f"{path.name}: values outside uint8 range")
    print(f"{args.npz} is valid: {len(keys)} images")


if __name__ == "__main__":
    main()

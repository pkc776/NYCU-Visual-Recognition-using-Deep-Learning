from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


@dataclass(frozen=True)
class Pair:
    degraded: Path
    clean: Path
    kind: str
    index: int


def _parse_degraded_name(path: Path) -> tuple[str, int]:
    stem = path.stem
    kind, idx = stem.split("-")
    return kind, int(idx)


def collect_pairs(root: str | Path) -> list[Pair]:
    root = Path(root)
    degraded_dir = root / "train" / "degraded"
    clean_dir = root / "train" / "clean"
    pairs: list[Pair] = []
    for degraded in sorted(degraded_dir.glob("*.png")):
        kind, idx = _parse_degraded_name(degraded)
        clean = clean_dir / f"{kind}_clean-{idx}.png"
        if not clean.exists():
            raise FileNotFoundError(f"Missing clean target for {degraded}: {clean}")
        pairs.append(Pair(degraded=degraded, clean=clean, kind=kind, index=idx))
    if not pairs:
        raise RuntimeError(f"No training pairs found under {root}")
    return pairs


def stratified_split(
    pairs: list[Pair], val_per_degradation: int = 100, seed: int = 777
) -> tuple[list[Pair], list[Pair]]:
    by_kind: dict[str, list[Pair]] = {}
    for pair in pairs:
        by_kind.setdefault(pair.kind, []).append(pair)

    rng = random.Random(seed)
    train: list[Pair] = []
    val: list[Pair] = []
    for kind in sorted(by_kind):
        items = by_kind[kind][:]
        rng.shuffle(items)
        val.extend(sorted(items[:val_per_degradation], key=lambda p: p.index))
        train.extend(sorted(items[val_per_degradation:], key=lambda p: p.index))
    return train, val


def _paired_augment(degraded: Image.Image, clean: Image.Image) -> tuple[Image.Image, Image.Image]:
    import random
    import torchvision.transforms.functional as TF
    # Only Horizontal Flip is safe. Rain/Snow falls downwards (gravity)! 
    # vflip makes rain fall upwards, rotate 90 makes it horizontal.
    if random.random() < 0.5:
        degraded = TF.hflip(degraded)
        clean = TF.hflip(clean)
    return degraded, clean



def _paired_crop(degraded: Image.Image, clean: Image.Image, crop_size: int) -> tuple[Image.Image, Image.Image]:
    w, h = degraded.size
    if w <= crop_size or h <= crop_size:
        return degraded, clean
    
    i = __import__('random').randint(0, h - crop_size)
    j = __import__('random').randint(0, w - crop_size)
    
    import torchvision.transforms.functional as TF
    degraded = TF.crop(degraded, i, j, crop_size, crop_size)
    clean = TF.crop(clean, i, j, crop_size, crop_size)
    return degraded, clean

class RestorationDataset(Dataset):
    def __init__(self, pairs: list[Pair], augment: bool = False, crop_size: int = 256):
        self.pairs = pairs
        self.augment = augment
        self.crop_size = crop_size

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        pair = self.pairs[idx]
        degraded = Image.open(pair.degraded).convert("RGB")
        clean = Image.open(pair.clean).convert("RGB")
        if degraded.size != clean.size:
            raise ValueError(f"Size mismatch: {pair.degraded} {degraded.size} vs {pair.clean} {clean.size}")
        
        if self.augment:
            degraded, clean = _paired_crop(degraded, clean, self.crop_size)
            degraded, clean = _paired_augment(degraded, clean)

        return {
            "input": TF.to_tensor(degraded),
            "target": TF.to_tensor(clean),
            "kind": pair.kind,
            "name": pair.degraded.name,
        }


class TestDataset(Dataset):
    def __init__(self, test_dir: str | Path):
        self.files = sorted(Path(test_dir).glob("*.png"), key=lambda p: int(p.stem))
        if not self.files:
            raise RuntimeError(f"No test images found under {test_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        path = self.files[idx]
        image = Image.open(path).convert("RGB")
        return {"input": TF.to_tensor(image), "name": path.name}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="release_folder/hw4_realse_dataset")
    parser.add_argument("--val-per-degradation", type=int, default=100)
    args = parser.parse_args()

    pairs = collect_pairs(args.root)
    train, val = stratified_split(pairs, args.val_per_degradation)
    sizes = set()
    for pair in pairs:
        with Image.open(pair.degraded) as image:
            sizes.add(image.size)
    by_kind = {kind: sum(pair.kind == kind for pair in pairs) for kind in sorted({p.kind for p in pairs})}
    print(f"pairs={len(pairs)} train={len(train)} val={len(val)} by_kind={by_kind} sizes={sorted(sizes)}")


if __name__ == "__main__":
    main()

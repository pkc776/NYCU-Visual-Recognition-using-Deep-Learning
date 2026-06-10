from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import TestDataset
from .model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/promptir_hw4.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-dir", default="")
    parser.add_argument("--output", default="pred.npz")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--tta", action="store_true")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_transform(x: torch.Tensor, mode: int) -> torch.Tensor:
    if mode == 0:
        return x
    if mode == 1:
        return torch.flip(x, dims=(-1,))
    if mode == 2:
        return torch.flip(x, dims=(-2,))
    if mode == 3:
        return torch.transpose(x, -1, -2)
    if mode == 4:
        return torch.flip(torch.transpose(x, -1, -2), dims=(-1,))
    if mode == 5:
        return torch.flip(torch.transpose(x, -1, -2), dims=(-2,))
    if mode == 6:
        return torch.rot90(x, 1, dims=(-2, -1))
    if mode == 7:
        return torch.rot90(x, 3, dims=(-2, -1))
    raise ValueError(mode)


def invert_transform(x: torch.Tensor, mode: int) -> torch.Tensor:
    if mode == 0:
        return x
    if mode == 1:
        return torch.flip(x, dims=(-1,))
    if mode == 2:
        return torch.flip(x, dims=(-2,))
    if mode == 3:
        return torch.transpose(x, -1, -2)
    if mode == 4:
        return torch.transpose(torch.flip(x, dims=(-1,)), -1, -2)
    if mode == 5:
        return torch.transpose(torch.flip(x, dims=(-2,)), -1, -2)
    if mode == 6:
        return torch.rot90(x, 3, dims=(-2, -1))
    if mode == 7:
        return torch.rot90(x, 1, dims=(-2, -1))
    raise ValueError(mode)


@torch.no_grad()
def predict(model: torch.nn.Module, x: torch.Tensor, tta: bool) -> torch.Tensor:
    if not tta:
        return model(x)
    preds = []
    # Rain and snow have a gravity-aligned direction; only horizontal flip is safe.
    for mode in (0, 1):
        y = model(apply_transform(x, mode))
        preds.append(invert_transform(y, mode))
    return torch.stack(preds, dim=0).mean(dim=0)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    test_dir = args.test_dir or str(Path(cfg["data_root"]) / "test" / "degraded")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TestDataset(test_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_cfg = checkpoint.get("config", cfg).get("model", cfg.get("model", {}))
    model = build_model(model_cfg).to(device)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=True)
    model.eval()

    pred_dict: dict[str, np.ndarray] = {}
    for batch in tqdm(loader, desc="infer"):
        image = batch["input"].to(device, non_blocking=True)
        names = batch["name"]
        pred = predict(model, image, args.tta).clamp(0, 1)
        pred = (pred * 255.0).round().to(torch.uint8).cpu().numpy()
        for name, array in zip(names, pred):
            pred_dict[name] = array

    expected = {f"{i}.png" for i in range(100)}
    missing = sorted(expected - set(pred_dict), key=lambda x: int(Path(x).stem))
    if missing:
        raise RuntimeError(f"Missing test predictions: {missing[:10]}")
    np.savez(args.output, **{k: pred_dict[k] for k in sorted(pred_dict, key=lambda x: int(Path(x).stem))})
    print(f"saved {len(pred_dict)} images to {args.output}")


if __name__ == "__main__":
    main()

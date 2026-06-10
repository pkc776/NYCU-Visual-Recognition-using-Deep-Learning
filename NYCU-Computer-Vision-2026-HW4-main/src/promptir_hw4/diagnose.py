from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import functional as TF

from .data import RestorationDataset, collect_pairs, stratified_split
from .metrics import psnr, ssim
from .model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect paired restoration data and checkpoint outputs.")
    parser.add_argument("--config", default="configs/promptir_hw4.yaml")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--output-dir", default="outputs/diagnostics")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--first-n", action="store_true", help="Use the first N samples instead of random samples.")
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    x = x.detach().cpu().clamp(0, 1)
    return TF.to_pil_image(x)


def make_sheet(
    inputs: list[torch.Tensor],
    preds: list[torch.Tensor] | None,
    targets: list[torch.Tensor],
    names: list[str],
    output_path: Path,
) -> None:
    rows = len(inputs)
    cols = 3 if preds is not None else 2
    tile_w, tile_h = inputs[0].shape[-1], inputs[0].shape[-2]
    label_h = 18
    sheet = Image.new("RGB", (cols * tile_w, rows * (tile_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    headers = ["input", "output", "target"] if preds is not None else ["input", "target"]

    for row in range(rows):
        row_images = [inputs[row]]
        if preds is not None:
            row_images.append(preds[row])
        row_images.append(targets[row])
        for col, image_tensor in enumerate(row_images):
            x0 = col * tile_w
            y0 = row * (tile_h + label_h)
            label = f"{headers[col]} {names[row]}" if col == 0 else headers[col]
            draw.text((x0 + 4, y0 + 2), label, fill=(0, 0, 0))
            sheet.paste(tensor_to_pil(image_tensor), (x0, y0 + label_h))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def load_model(checkpoint_path: str, cfg: dict[str, Any], device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_cfg = checkpoint.get("config", cfg).get("model", cfg.get("model", {}))
    model = build_model(model_cfg).to(device)
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    model.eval()
    return model


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    train_cfg = cfg["train"]
    pairs = collect_pairs(cfg["data_root"])
    train_pairs, val_pairs = stratified_split(
        pairs,
        val_per_degradation=int(train_cfg["val_per_degradation"]),
        seed=int(cfg.get("seed", 777)),
    )
    selected_pairs = train_pairs if args.split == "train" else val_pairs
    num_samples = min(args.num_samples, len(selected_pairs))
    if args.first_n:
        indices = list(range(num_samples))
    else:
        rng = random.Random(args.seed)
        indices = sorted(rng.sample(range(len(selected_pairs)), num_samples))

    dataset = RestorationDataset(selected_pairs, augment=False)
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, cfg, device) if args.checkpoint else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_inputs: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_preds: list[torch.Tensor] = []
    all_names: list[str] = []
    input_psnr_values: list[torch.Tensor] = []
    input_ssim_values: list[torch.Tensor] = []
    pred_psnr_values: list[torch.Tensor] = []
    pred_ssim_values: list[torch.Tensor] = []

    for batch in loader:
        image = batch["input"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        pred = model(image).clamp(0, 1) if model is not None else None

        input_psnr_values.append(psnr(image, target).cpu())
        input_ssim_values.append(ssim(image, target).cpu())
        if pred is not None:
            pred_psnr_values.append(psnr(pred, target).cpu())
            pred_ssim_values.append(ssim(pred, target).cpu())

        all_inputs.extend(image.cpu())
        all_targets.extend(target.cpu())
        if pred is not None:
            all_preds.extend(pred.cpu())
        all_names.extend(batch["name"])

    input_psnr_mean = torch.cat(input_psnr_values).mean().item()
    input_ssim_mean = torch.cat(input_ssim_values).mean().item()
    pred_psnr_mean = torch.cat(pred_psnr_values).mean().item() if pred_psnr_values else None
    pred_ssim_mean = torch.cat(pred_ssim_values).mean().item() if pred_ssim_values else None

    sheet_path = output_dir / f"{args.split}_pairs.png"
    make_sheet(
        all_inputs,
        all_preds if model is not None else None,
        all_targets,
        all_names,
        sheet_path,
    )

    stats_path = output_dir / f"{args.split}_stats.txt"
    lines = [
        f"split={args.split}",
        f"indices={indices}",
        f"names={all_names}",
        f"input_psnr={input_psnr_mean:.4f}",
        f"input_ssim={input_ssim_mean:.4f}",
    ]
    if pred_psnr_mean is not None and pred_ssim_mean is not None:
        lines.extend(
            [
                f"output_psnr={pred_psnr_mean:.4f}",
                f"output_ssim={pred_ssim_mean:.4f}",
                f"delta_psnr={pred_psnr_mean - input_psnr_mean:.4f}",
                f"delta_ssim={pred_ssim_mean - input_ssim_mean:.4f}",
            ]
        )
    stats_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"wrote {sheet_path}")
    print(f"wrote {stats_path}")


if __name__ == "__main__":
    main()

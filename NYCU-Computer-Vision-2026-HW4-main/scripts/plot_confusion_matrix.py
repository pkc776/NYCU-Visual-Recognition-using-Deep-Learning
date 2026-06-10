from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from eval_route_kinds import predict_snow_prob, train_kind_classifier
from src.promptir_hw4.data import collect_pairs, stratified_split


KINDS = ("rain", "snow")


def confusion_matrix(true_kinds: list[str], pred_kinds: list[str]) -> np.ndarray:
    index = {kind: i for i, kind in enumerate(KINDS)}
    matrix = np.zeros((len(KINDS), len(KINDS)), dtype=np.int64)
    for true, pred in zip(true_kinds, pred_kinds):
        matrix[index[true], index[pred]] += 1
    return matrix


def save_csv(matrix: np.ndarray, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred", *KINDS])
        for kind, row in zip(KINDS, matrix):
            writer.writerow([kind, *row.tolist()])


def save_plot(matrix: np.ndarray, path: Path, split: str) -> None:
    row_sums = matrix.sum(axis=1, keepdims=True).clip(min=1)
    normalized = matrix / row_sums

    cell = 135
    left = 145
    top = 100
    right = 35
    bottom = 70
    width = left + cell * len(KINDS) + right
    height = top + cell * len(KINDS) + bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    title = f"{split.capitalize()} degradation confusion matrix"
    draw.text((left, 24), title, fill=(0, 0, 0), font=font)
    draw.text((left + 35, height - 35), "Predicted degradation", fill=(0, 0, 0), font=font)
    draw.text((12, top + cell // 2), "Ground truth", fill=(0, 0, 0), font=font)

    for j, kind in enumerate(KINDS):
        x = left + j * cell + cell // 2 - 12
        draw.text((x, top - 28), kind, fill=(0, 0, 0), font=font)
    for i, kind in enumerate(KINDS):
        y = top + i * cell + cell // 2 - 6
        draw.text((left - 58, y), kind, fill=(0, 0, 0), font=font)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            rate = float(normalized[i, j])
            blue = int(255 - 145 * rate)
            fill = (blue, blue, 255)
            x0 = left + j * cell
            y0 = top + i * cell
            x1 = x0 + cell
            y1 = y0 + cell
            draw.rectangle((x0, y0, x1, y1), fill=fill, outline=(80, 80, 80))
            text = f"{matrix[i, j]}\n{rate * 100:.1f}%"
            color = "white" if rate > 0.55 else "black"
            bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=5)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.multiline_text(
                (x0 + (cell - tw) / 2, y0 + (cell - th) / 2),
                text,
                fill=color,
                font=font,
                spacing=5,
                align="center",
            )

    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/promptir_hw4.yaml")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", default="outputs/diagnostics")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    pairs = collect_pairs(cfg["data_root"])
    train_pairs, val_pairs = stratified_split(
        pairs,
        val_per_degradation=int(cfg["train"]["val_per_degradation"]),
        seed=int(cfg.get("seed", 777)),
    )
    eval_pairs = train_pairs if args.split == "train" else val_pairs

    mean, std, w, b = train_kind_classifier(train_pairs)
    probs = predict_snow_prob([pair.degraded for pair in eval_pairs], mean, std, w, b)
    pred_kinds = ["snow" if prob >= args.threshold else "rain" for prob in probs]
    true_kinds = [pair.kind for pair in eval_pairs]
    matrix = confusion_matrix(true_kinds, pred_kinds)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"confusion_matrix_{args.split}"
    save_csv(matrix, output_dir / f"{stem}.csv")
    save_plot(matrix, output_dir / f"{stem}.png", args.split)

    accuracy = sum(t == p for t, p in zip(true_kinds, pred_kinds)) / max(1, len(true_kinds))
    print(f"split={args.split} threshold={args.threshold} accuracy={accuracy:.4f}")
    print(matrix)
    print(f"saved {output_dir / f'{stem}.png'}")
    print(f"saved {output_dir / f'{stem}.csv'}")


if __name__ == "__main__":
    main()

"""
inference.py – Generate pred.json for the test set.

Usage:
    python -m deformable_detr_hw2.inference \
        --test_images nycu-hw2-data/test \
        --checkpoint checkpoints/best_map.pth \
        --output_json pred.json \
        --score_threshold 0.3 \
        --nms_iou_threshold 0.5
"""

import argparse
import json
from typing import List

import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from .data import TestDataset, build_val_transforms, test_collate_fn
    from .modeling import build_model
    from .postprocess import DeformableDetrPostProcessor
except ImportError:
    # Fallback for direct execution (e.g. python inference.py)
    from data import TestDataset, build_val_transforms, test_collate_fn
    from modeling import build_model
    from postprocess import DeformableDetrPostProcessor


def pad_images(imgs: List[torch.Tensor]):
    """Pad a list of CHW tensors to max spatial size → (B, C, H, W)."""
    C = imgs[0].shape[0]
    max_h = max(x.shape[1] for x in imgs)
    max_w = max(x.shape[2] for x in imgs)
    batch = torch.zeros(
        len(imgs), C, max_h, max_w, dtype=imgs[0].dtype, device=imgs[0].device
    )
    mask = torch.zeros(
        len(imgs),
        max_h,
        max_w,
        dtype=torch.long,
        device=imgs[0].device)
    for i, img in enumerate(imgs):
        batch[i, :, : img.shape[1], : img.shape[2]] = img
        mask[i, : img.shape[1], : img.shape[2]] = 1
    return batch, mask


def parse_args():
    p = argparse.ArgumentParser("Deformable DETR Inference")
    p.add_argument("--test_images", default="nycu-hw2-data/test")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_json", default="pred.json")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_image_size", type=int, default=800)
    p.add_argument("--image_max_size", type=int, default=1333)
    p.add_argument("--score_threshold", type=float, default=0.3)
    p.add_argument("--nms_iou_threshold", type=float, default=0.5)
    p.add_argument("--max_detections", type=int, default=300)
    p.add_argument("--amp", action="store_true")
    # Model args (must match training)
    p.add_argument("--num_queries", type=int, default=300)
    p.add_argument("--num_encoder_layers", type=int, default=6)
    p.add_argument("--num_decoder_layers", type=int, default=6)
    p.add_argument("--two_stage", action="store_true")
    p.add_argument("--num_feature_levels", type=int, default=4)
    p.add_argument(
        "--device",
        default="cuda",
        help="Device to run inference on (e.g., 'cuda', 'cuda:0', 'cpu')",
    )
    return p.parse_args()


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    test_loader: DataLoader,
    postprocessor: DeformableDetrPostProcessor,
    device: torch.device,
    val_image_size: int,
    amp: bool = False,
) -> List[dict]:
    model.eval()
    all_results = []

    for imgs, image_ids, w0s, h0s in tqdm(test_loader, desc="Inference"):
        imgs = [x.to(device) for x in imgs]
        pixel_values, pixel_mask = pad_images(imgs)

        with autocast(enabled=amp):
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        logits = outputs.logits
        pred_boxes = outputs.pred_boxes

        # Since model processes with pixel_mask, pred_boxes are normalized
        # relative to the unpadded valid shapes. Since the val resize is
        # uniform, the normalized coordinates are effectively aligned with
        # the original image's dimensions. We can pass the original sizes
        # directly!
        orig_sizes = torch.tensor(
            [[h0s[i], w0s[i]] for i in range(len(imgs))],
            dtype=torch.long,
            device=device,
        )

        preds = postprocessor(logits, pred_boxes, orig_sizes, list(image_ids))
        all_results.extend(preds)

    return all_results


def main():
    args = parse_args()
    device = torch.device(
        args.device if (
            args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"Device: {device}")

    # ── Model ────────────────────────────────────────────────────────────────
    print("Building model …")
    model = build_model(
        num_queries=args.num_queries,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        pretrained_backbone=False,  # we load from checkpoint
        two_stage=args.two_stage,
        num_feature_levels=args.num_feature_levels,
    )
    print(f"Loading checkpoint: {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location=device)
    model_state = state.get("model", state)
    model.load_state_dict(model_state, strict=True)
    model.to(device)

    # ── Data ─────────────────────────────────────────────────────────────────
    tf = build_val_transforms(val_longer=args.val_image_size)
    test_ds = TestDataset(args.test_images, transforms=tf)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=test_collate_fn,
        pin_memory=True,
    )
    print(f"Test images: {len(test_ds)}")

    # ── Postprocessor ───────────────────────────────────────────────────────
    postprocessor = DeformableDetrPostProcessor(
        score_threshold=args.score_threshold,
        nms_iou_threshold=args.nms_iou_threshold,
        max_detections=args.max_detections,
    )

    # ── Inference ───────────────────────────────────────────────────────────
    results = run_inference(
        model,
        test_loader,
        postprocessor,
        device,
        val_image_size=args.val_image_size,
        amp=args.amp,
    )
    print(f"Total detections: {len(results)}")

    with open(args.output_json, "w") as f:
        json.dump(results, f)
    print(f"Saved predictions to: {args.output_json}")


if __name__ == "__main__":
    main()

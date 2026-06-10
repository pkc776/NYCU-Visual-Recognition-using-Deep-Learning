"""
train.py – Main training entry point for Deformable DETR digit detection.

Run:
    python -m deformable_detr_hw2.train --data_root nycu-hw2-data --epochs 50 \
        --batch_size 4 --lr 2e-4 --lr_backbone 2e-5 --amp --wandb \
        --wandb_project nycu-vrdl-hw2 --wandb_run_name exp01
"""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

# ── local imports ───────────────────────────────────────────────────────
from .data import (
    DigitDetectionDataset,
    build_train_transforms,
    build_val_transforms,
    collate_fn,
)
from .modeling import build_model
from .postprocess import DeformableDetrPostProcessor
from .evaluate import CocoEvaluator
from .utils import (
    build_optimizer,
    build_scheduler,
    MetricLogger,
    save_checkpoint,
    load_checkpoint,
    JsonLogger,
)

# ─── argument parsing ───────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser("Deformable DETR Digit Detector")

    # Data
    p.add_argument("--data_root", default="nycu-hw2-data",
                   help="Root directory of the dataset.")
    p.add_argument(
        "--output_dir",
        default="checkpoints",
        help="Directory to save checkpoints and logs.",
    )

    # Training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Per-GPU micro-batch size. Use --grad_accum to maintain effective batch.",
    )
    p.add_argument(
        "--grad_accum",
        type=int,
        default=2,
        help="Gradient accumulation steps. Effective batch = batch_size * grad_accum.",
    )
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr_backbone", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--clip_grad_norm", type=float, default=0.1)
    p.add_argument(
        "--amp",
        action="store_true",
        help="Use mixed precision (bfloat16 preferred).")
    p.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to trade compute for memory.",
    )
    p.add_argument(
        "--resume",
        default="",
        help="Path to checkpoint to resume.")
    p.add_argument("--seed", type=int, default=42)

    # Model
    p.add_argument("--num_queries", type=int, default=300)
    p.add_argument("--num_encoder_layers", type=int, default=6)
    p.add_argument("--num_decoder_layers", type=int, default=6)
    p.add_argument("--two_stage", action="store_true")
    p.add_argument("--num_feature_levels", type=int, default=4)

    # Image sizes are HARDCODED in data.py based on dataset analysis:
    #   TRAIN_SCALES = [192, 224, 256, 288, 320] (longer side)
    #   VAL_LONGER   = 320
    # No need to pass them as args.

    # Evaluation
    p.add_argument(
        "--eval_interval",
        type=int,
        default=1,
        help="Run mAP evaluation every N epochs (0 = never).",
    )
    p.add_argument("--score_threshold", type=float, default=0.3)
    p.add_argument("--nms_iou_threshold", type=float, default=0.5)

    # Speed
    p.add_argument(
        "--max_steps_per_epoch",
        type=int,
        default=0,
        help="Cap training steps per epoch (0 = no cap).",
    )
    p.add_argument(
        "--max_eval_batches",
        type=int,
        default=0,
        help="Cap eval batches (0 = full val).",
    )

    # Logging
    p.add_argument(
        "--log_interval",
        type=int,
        default=50,
        help="Print training stats every N steps.",
    )
    p.add_argument("--no_tqdm", action="store_true", help="Disable tqdm.")

    # WandB
    p.add_argument("--wandb", action="store_true", help="Enable W&B logging.")
    p.add_argument("--wandb_project", default="nycu-vrdl-hw2-deformable-detr")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument(
        "--wandb_mode",
        default="online",
        choices=["online", "offline", "disabled"],
        help="W&B mode.",
    )

    return p.parse_args()


# ─── helpers ────────────────────────────────────────────────────────────


def set_seed(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_device(imgs: List[torch.Tensor],
              targets: List[dict],
              device: torch.device):
    imgs = [x.to(device) for x in imgs]
    targets = [
        {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
        for t in targets
    ]
    return imgs, targets


def pad_images(imgs: List[torch.Tensor]):
    """
    Pad a list of CHW tensors of possibly different sizes to the same
    (max_H, max_W), filling with zeros (ImageNet-normalized mean ≈ 0).
    Returns a (B, C, H, W) tensor and a (B, H, W) pixel_mask.
    """
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


def prepare_hf_labels(targets: List[dict], device: torch.device):
    """
    Convert our target dicts to HuggingFace Deformable DETR label format.

    HF expects a list of dicts with keys 'class_labels' and 'boxes'.
    boxes must be (cx, cy, w, h) normalized floats.
    class_labels are long tensors of label indices.
    """
    hf_labels = []
    for t in targets:
        hf_labels.append(
            {
                "class_labels": t["labels"].to(device),
                "boxes": t["boxes"].to(device),
            }
        )
    return hf_labels


# ─── evaluation loop ────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    postprocessor: DeformableDetrPostProcessor,
    evaluator: CocoEvaluator,
    device: torch.device,
    amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
    max_batches: int = 0,
) -> Dict[str, float]:
    model.eval()
    evaluator.reset()

    for i, (imgs, targets) in enumerate(val_loader):
        if max_batches > 0 and i >= max_batches:
            break

        imgs, targets = to_device(imgs, targets, device)
        pixel_values, pixel_mask = pad_images(imgs)

        with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        logits = outputs.logits.float()  # (B, Q, C)
        pred_boxes = outputs.pred_boxes.float()  # (B, Q, 4)  normalized cxcywh

        # orig sizes for un-normalizing
        orig_sizes = torch.stack([t["orig_size"]
                                 for t in targets], dim=0).to(device)
        image_ids = [t["image_id"] for t in targets]

        preds = postprocessor(logits, pred_boxes, orig_sizes, image_ids)
        evaluator.update(preds)

    metrics = evaluator.summarize()
    return metrics


# ─── training loop ──────────────────────────────────────────────────────


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    train_loader: DataLoader,
    device: torch.device,
    epoch: int,
    scaler: Optional[GradScaler],
    amp_dtype: torch.dtype,
    use_amp: bool,
    args,
    wandb_run=None,
) -> Dict[str, float]:
    model.train()
    logger = MetricLogger()
    grad_accum = getattr(args, "grad_accum", 1)

    total_steps = len(train_loader)
    if args.max_steps_per_epoch > 0:
        total_steps = min(total_steps, args.max_steps_per_epoch)

    try:
        from tqdm import tqdm

        use_tqdm = not args.no_tqdm
    except ImportError:
        use_tqdm = False

    it = iter(train_loader)
    pbar = (
        tqdm(range(total_steps), desc=f"Epoch {epoch}")
        if use_tqdm
        else range(total_steps)
    )

    global_step = (epoch - 1) * total_steps
    optimizer.zero_grad()

    for step_i in pbar:
        try:
            imgs, targets = next(it)
        except StopIteration:
            break

        imgs, targets = to_device(imgs, targets, device)
        pixel_values, pixel_mask = pad_images(imgs)
        labels = prepare_hf_labels(targets, device)

        # ── Forward ─────────────────────────────────────────────────────────
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            outputs = model(
                pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels
            )
            # Scale loss for gradient accumulation
            loss = outputs.loss / grad_accum

        # ── Backward ────────────────────────────────────────────────────────
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # ── Optimizer step every `grad_accum` micro-batches ─────────────────
        is_update_step = (
            (step_i + 1) %
            grad_accum == 0) or (
            step_i + 1 == total_steps)
        if is_update_step:
            if scaler is not None:
                if args.clip_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        model.parameters(), args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                if args.clip_grad_norm > 0:
                    nn.utils.clip_grad_norm_(
                        model.parameters(), args.clip_grad_norm)
                optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        # ── Logging (use un-scaled loss for display) ─────────────────────────
        loss_val = loss.item() * grad_accum  # restore scale for display
        loss_dict = {}
        if outputs.loss_dict is not None:
            loss_dict = {k: v.item() for k, v in outputs.loss_dict.items()}
        loss_dict["loss"] = loss_val
        logger.update(**loss_dict)

        if use_tqdm:
            pbar.set_postfix({"loss": f"{loss_val:.4f}"})

        if (step_i + 1) % args.log_interval == 0:
            lr_now = scheduler.get_last_lr()[0] if scheduler else args.lr
            print(
                f"[Epoch {epoch:3d}  Step {step_i+1:4d}/{total_steps}]  "
                f"{logger}  lr={lr_now:.2e}",
                flush=True,
            )

        # WandB step-level logging
        if wandb_run is not None:
            log_dict = {f"train/{k}": v for k, v in loss_dict.items()}
            log_dict["train/lr"] = scheduler.get_last_lr()[0] if scheduler else args.lr
            log_dict["epoch"] = epoch
            wandb_run.log(log_dict, step=global_step + step_i)

    return logger.global_avg()


# ─── main ───────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)

    # Reduce CUDA memory fragmentation (recommended for large models)
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True")

    print(f"Device: {device}")
    eff_batch = args.batch_size * args.grad_accum
    print(
        f"Micro-batch: {args.batch_size}  Grad-accum: {args.grad_accum}  "
        f"Effective batch: {eff_batch}"
    )
    print(f"Args: {vars(args)}")

    # ── WandB ───────────────────────────────────────────────────────────────
    wandb_run = None
    if args.wandb:
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args),
                mode=args.wandb_mode,
                dir=args.output_dir,
            )
            print(f"[WandB] Run URL: {wandb_run.url}")
        except Exception as e:
            print(f"[WandB] Failed to init: {e}. Continuing without W&B.")
            wandb_run = None

    # ── Data ─────────────────────────────────────────────────────────────────
    data_root = Path(args.data_root)

    # Image sizes are hardcoded in data.py (TRAIN_SCALES, VAL_LONGER)
    # based on dataset analysis: longer-side median=104, P75=158, max=876
    train_tf = build_train_transforms()
    val_tf = build_val_transforms()

    train_ds = DigitDetectionDataset(
        img_dir=str(data_root / "train"),
        ann_file=str(data_root / "train.json"),
        transforms=train_tf,
    )
    val_ds = DigitDetectionDataset(
        img_dir=str(data_root / "valid"),
        ann_file=str(data_root / "valid.json"),
        transforms=val_tf,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        # keep workers alive between epochs
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    print(f"Train: {len(train_ds)} images | Val: {len(val_ds)} images")

    # ── Model ────────────────────────────────────────────────────────────────
    print("Building model …")
    model = build_model(
        num_queries=args.num_queries,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        pretrained_backbone=True,
        two_stage=args.two_stage,
        num_feature_levels=args.num_feature_levels,
    )
    model.to(device)

    # Gradient checkpointing: recomputes activations during backward to save
    # VRAM
    if getattr(args, "gradient_checkpointing", False):
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            print("[Memory] Gradient checkpointing enabled.")
        else:
            print(
                "[Memory] gradient_checkpointing_enable() not available on this model."
            )

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Parameters: {n_params/1e6:.1f}M total, {n_trainable/1e6:.1f}M trainable")

    # ── Optimizer & Scheduler ───────────────────────────────────────────────
    optimizer = build_optimizer(
        model,
        lr=args.lr,
        lr_backbone=args.lr_backbone,
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = len(train_loader)
    if args.max_steps_per_epoch > 0:
        steps_per_epoch = min(steps_per_epoch, args.max_steps_per_epoch)

    scheduler = build_scheduler(
        optimizer, epochs=args.epochs, steps_per_epoch=steps_per_epoch
    )

    scaler: Optional[GradScaler] = None
    amp_dtype = torch.bfloat16
    if args.amp and device.type == "cuda":
        # Try bfloat16 first (more numerically stable, same range as float32)
        if torch.cuda.is_bf16_supported():
            print("[AMP] Using bfloat16 (stable).")
            amp_dtype = torch.bfloat16
        else:
            print("[AMP] GPU doesn't support bfloat16, using float16 with GradScaler.")
            amp_dtype = torch.float16
            scaler = GradScaler(device="cuda")

    # ── Resume ──────────────────────────────────────────────────────────────
    start_epoch = 1
    best_map = 0.0
    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from {args.resume}")
        start_epoch_ckpt, best_map, _ = load_checkpoint(
            args.resume, model, optimizer, scheduler, device
        )
        start_epoch = start_epoch_ckpt + 1

    # ── Evaluation helpers ──────────────────────────────────────────────────
    postprocessor = DeformableDetrPostProcessor(
        score_threshold=args.score_threshold,
        nms_iou_threshold=args.nms_iou_threshold,
        max_detections=300,
    )
    evaluator = CocoEvaluator(ann_file=str(data_root / "valid.json"))
    json_log = JsonLogger(os.path.join(args.output_dir, "log.json"))

    # ── Training ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Starting training for {args.epochs} epochs")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        # ── Train ────────────────────────────────────────────────────────────
        train_metrics = train_one_epoch(
            model,
            optimizer,
            scheduler,
            train_loader,
            device,
            epoch,
            scaler,
            amp_dtype,
            args.amp,
            args,
            wandb_run=wandb_run,
        )

        elapsed = time.time() - t0
        print(
            f"\n[Epoch {epoch}/{args.epochs}]  "
            f"loss={train_metrics.get('loss', 0):.4f}  "
            f"time={elapsed:.1f}s"
        )

        # ── Validation ───────────────────────────────────────────────────────
        val_metrics: Dict[str, float] = {}
        do_eval = args.eval_interval > 0 and (
            epoch % args.eval_interval == 0 or epoch == args.epochs
        )
        if do_eval:
            print("  Running mAP evaluation …", flush=True)
            val_metrics = evaluate(
                model,
                val_loader,
                postprocessor,
                evaluator,
                device,
                amp=args.amp,
                amp_dtype=amp_dtype,
                max_batches=args.max_eval_batches,
            )
            print(
                f"  mAP={val_metrics['mAP']:.4f}  "
                f"AP50={val_metrics['AP50']:.4f}  "
                f"AP75={val_metrics['AP75']:.4f}",
                flush=True,
            )

            # Save best model
            if val_metrics["mAP"] > best_map:
                best_map = val_metrics["mAP"]
                save_checkpoint(
                    args.output_dir,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_map,
                    val_metrics,
                    name="best_map.pth",
                )
                print(f"  ✓ New best mAP={best_map:.4f} → saved best_map.pth")

        # Always save last checkpoint
        save_checkpoint(
            args.output_dir,
            model,
            optimizer,
            scheduler,
            epoch,
            best_map,
            val_metrics,
            name="last.pth",
        )

        # ── Logging ──────────────────────────────────────────────────────────
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "best_map": best_map,
            "time_s": elapsed,
        }
        json_log.log(record)

        if wandb_run is not None:
            log_dict = {
                "epoch": epoch,
                "train/epoch_loss": train_metrics.get("loss", 0),
            }
            if val_metrics:
                for k, v in val_metrics.items():
                    log_dict[f"val/{k}"] = v
                log_dict["val/best_mAP"] = best_map
            wandb_run.log(log_dict, step=epoch * steps_per_epoch)

    print(f"\nTraining complete. Best val mAP = {best_map:.4f}")
    print(f"Checkpoints saved to: {args.output_dir}")

    if wandb_run is not None:
        wandb_run.summary["best_mAP"] = best_map
        wandb_run.finish()


if __name__ == "__main__":
    main()

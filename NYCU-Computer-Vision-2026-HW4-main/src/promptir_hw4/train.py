from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
import yaml
from PIL import Image, ImageDraw
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.transforms import functional as TF
from tqdm import tqdm

from .data import RestorationDataset, collect_pairs, stratified_split
from .ema import ModelEMA
from .losses import RestorationLoss
from .metrics import AverageMeter, psnr, ssim
from .model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/promptir_hw4.yaml")
    parser.add_argument("--resume", default="")
    parser.add_argument("--init-checkpoint", default="", help="Load model weights only and start a fresh training schedule.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--limit-train-batches", type=int, default=0)
    parser.add_argument("--limit-val-batches", type=int, default=0)
    parser.add_argument("--overfit-samples", type=int, default=0)
    parser.add_argument("--dump-train-batch", default="")
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def is_dist() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def setup_distributed() -> tuple[int, int, int]:
    if not is_dist():
        return 0, 1, 0
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def seed_everything(seed: int, rank: int = 0) -> None:
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def cosine_lr(epoch: int, step: int, steps_per_epoch: int, cfg: dict[str, Any]) -> float:
    total_steps = cfg["epochs"] * steps_per_epoch
    warmup_steps = cfg["warmup_epochs"] * steps_per_epoch
    current = epoch * steps_per_epoch + step
    if current < warmup_steps:
        return cfg["lr"] * float(current + 1) / max(1, warmup_steps)
    progress = (current - warmup_steps) / max(1, total_steps - warmup_steps)
    return cfg["min_lr"] + 0.5 * (cfg["lr"] - cfg["min_lr"]) * (1.0 + math.cos(math.pi * progress))


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def reduce_mean(value: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size <= 1:
        return value
    value = value.clone()
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value / world_size


def save_pair_grid(batch: dict[str, Any], path: str | Path, max_samples: int = 8) -> None:
    inputs = batch["input"][:max_samples].detach().cpu().clamp(0, 1)
    targets = batch["target"][:max_samples].detach().cpu().clamp(0, 1)
    names = list(batch["name"])[: len(inputs)]
    tile_h, tile_w = inputs.shape[-2:]
    label_h = 18
    sheet = Image.new("RGB", (2 * tile_w, len(inputs) * (tile_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, (image, target, name) in enumerate(zip(inputs, targets, names)):
        y0 = row * (tile_h + label_h)
        draw.text((4, y0 + 2), f"input {name}", fill=(0, 0, 0))
        draw.text((tile_w + 4, y0 + 2), "target", fill=(0, 0, 0))
        sheet.paste(TF.to_pil_image(image), (0, y0 + label_h))
        sheet.paste(TF.to_pil_image(target), (tile_w, y0 + label_h))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    world_size: int,
    limit_batches: int = 0,
    rank: int = 0,
    desc: str = "val",
    log_images: bool = False,
    epoch: int = 0,
) -> dict[str, float]:
    model.eval()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    total = len(loader) if not limit_batches else min(len(loader), limit_batches)
    iterator = enumerate(tqdm(loader, total=total, disable=rank != 0, desc=desc, leave=False))
    
    vis_images = []

    for batch_idx, batch in iterator:
        if limit_batches and batch_idx >= limit_batches:
            break
        image = batch["input"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        pred = model(image)
        batch_psnr = psnr(pred, target).mean()
        batch_ssim = ssim(pred, target).mean()
        batch_psnr = reduce_mean(batch_psnr, world_size)
        batch_ssim = reduce_mean(batch_ssim, world_size)
        n = image.size(0) * world_size
        psnr_meter.update(float(batch_psnr.item()), n)
        ssim_meter.update(float(batch_ssim.item()), n)

        if log_images and rank == 0 and len(vis_images) < 4:
            # just take the first sample in batch
            in_np = image[0].cpu().numpy().transpose(1, 2, 0)
            target_np = target[0].cpu().numpy().transpose(1, 2, 0)
            pred_np = pred[0].cpu().numpy().transpose(1, 2, 0).clip(0, 1)
            
            combined = np.concatenate([in_np, pred_np, target_np], axis=1)
            vis_images.append(wandb.Image(combined, caption=f"Left: Input, Middle: Pred, Right: Target (Epoch {epoch+1})"))

    if log_images and rank == 0 and wandb.run is not None:
        wandb.log({f"{desc}_images": vis_images, "epoch": epoch})

    return {"psnr": psnr_meter.avg, "ssim": ssim_meter.avg}


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    best_psnr: float,
    cfg: dict[str, Any],
    ema: ModelEMA | None = None,
) -> None:
    raw_model = model.module if isinstance(model, DDP) else model
    payload = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_psnr": best_psnr,
        "config": cfg,
    }
    if ema is not None:
        payload["ema"] = ema.ema.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    train_cfg = cfg["train"]
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.num_workers is not None:
        train_cfg["num_workers"] = args.num_workers
    if args.grad_accum_steps is not None:
        train_cfg["grad_accum_steps"] = args.grad_accum_steps

    rank, world_size, local_rank = setup_distributed()
    seed_everything(int(cfg.get("seed", 777)), rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    output_dir = Path(cfg["output_dir"])
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        if not args.no_wandb:
            wandb.init(project="hw4-promptir", config=cfg, name="PromptIR_hw4", resume="allow")

    pairs = collect_pairs(cfg["data_root"])
    train_pairs, val_pairs = stratified_split(
        pairs,
        val_per_degradation=int(train_cfg["val_per_degradation"]),
        seed=int(cfg.get("seed", 777)),
    )
    if args.overfit_samples:
        train_pairs = train_pairs[: args.overfit_samples]
        val_pairs = train_pairs
    train_set = RestorationDataset(
        train_pairs,
        augment=bool(train_cfg.get("augment", True)) and not args.overfit_samples,
        crop_size=int(train_cfg.get("crop_size", 128)),
    )
    val_set = RestorationDataset(val_pairs, augment=False)
    train_sampler = DistributedSampler(train_set, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_set, shuffle=False) if world_size > 1 else None
    train_loader = DataLoader(
        train_set,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=True,
        drop_last=not bool(args.overfit_samples),
        persistent_workers=int(train_cfg["num_workers"]) > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(train_cfg["batch_size"]),
        sampler=val_sampler,
        num_workers=max(1, int(train_cfg["num_workers"]) // 2),
        pin_memory=True,
        persistent_workers=int(train_cfg["num_workers"]) > 1,
    )

    if args.dump_train_batch and rank == 0:
        save_pair_grid(next(iter(train_loader)), args.dump_train_batch)
        print(f"wrote train batch dump to {args.dump_train_batch}")

    model = build_model(cfg.get("model", {})).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    raw_model = model.module if isinstance(model, DDP) else model
    ema = ModelEMA(raw_model, decay=float(train_cfg["ema_decay"])) if train_cfg.get("ema_decay", 0) else None
    if ema is not None:
        ema.ema.to(device)

    criterion = RestorationLoss(
        l1_weight=float(train_cfg["l1_weight"]),
        ssim_weight=float(train_cfg["ssim_weight"]),
        edge_weight=float(train_cfg["edge_weight"]),
        residual_weight=float(train_cfg.get("residual_weight", 0.0)),
        freq_residual_weight=float(train_cfg.get("freq_residual_weight", 0.0)),
        snow_weight=float(train_cfg.get("snow_weight", 1.0)),
        hard_weight=float(train_cfg.get("hard_weight", 0.0)),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
        betas=(0.9, 0.99),
    )
    scaler = GradScaler(device.type, enabled=bool(train_cfg.get("amp", True)) and device.type == "cuda")

    start_epoch = 0
    best_psnr = -1.0
    if args.resume and args.init_checkpoint:
        raise ValueError("Use either --resume or --init-checkpoint, not both")
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=device)
        state = checkpoint.get("model", checkpoint)
        raw_model.load_state_dict(state)
        if ema is not None:
            ema_state = checkpoint.get("ema", state)
            ema.ema.load_state_dict(ema_state)
        if rank == 0:
            source_epoch = checkpoint.get("epoch", "unknown") if isinstance(checkpoint, dict) else "unknown"
            print(f"initialized model weights from {args.init_checkpoint} epoch={source_epoch}")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        raw_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_psnr = float(checkpoint.get("best_psnr", best_psnr))
        if ema is not None and "ema" in checkpoint:
            ema.ema.load_state_dict(checkpoint["ema"])

    if rank == 0:
        params = sum(p.numel() for p in raw_model.parameters()) / 1e6
        print(f"train={len(train_set)} val={len(val_set)} world_size={world_size} params={params:.2f}M")

    for epoch in range(start_epoch, int(train_cfg["epochs"])):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        loss_meter = AverageMeter()
        steps_per_epoch = len(train_loader) if not args.limit_train_batches else args.limit_train_batches
        grad_accum_steps = max(1, int(train_cfg.get("grad_accum_steps", 1)))
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(train_loader, disable=rank != 0, desc=f"epoch {epoch + 1}/{train_cfg['epochs']}")
        for step, batch in enumerate(progress):
            if args.limit_train_batches and step >= args.limit_train_batches:
                break
            lr = cosine_lr(epoch, step, steps_per_epoch, train_cfg)
            set_lr(optimizer, lr)

            image = batch["input"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            with autocast(device_type=device.type, enabled=scaler.is_enabled()):

                pred = model(image)
                loss, loss_parts = criterion(pred, target, image, batch.get("kind"))
            scaler.scale(loss / grad_accum_steps).backward()
            should_step = ((step + 1) % grad_accum_steps == 0) or ((step + 1) == steps_per_epoch)
            if should_step:
                if float(train_cfg.get("grad_clip", 0)) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip"]))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(raw_model)
            reduced_loss = reduce_mean(loss.detach(), world_size)
            loss_meter.update(float(reduced_loss.item()), image.size(0) * world_size)
            progress.set_postfix(loss=f"{loss_meter.avg:.4f}", lr=f"{lr:.2e}", accum=f"{(step % grad_accum_steps) + 1}/{grad_accum_steps}")
            
            if rank == 0 and wandb.run is not None and should_step:
                # Log batch metrics
                log_data = {"train/loss": reduced_loss.item(), "train/lr": lr}
                for k, v in loss_parts.items():
                    log_data[f"train/loss_{k}"] = v.item()
                wandb.log(log_data)

        if rank == 0 and wandb.run is not None:
            wandb.log({"train/epoch_loss": loss_meter.avg, "epoch": epoch})

        eval_every = int(train_cfg.get("eval_every", 10))
        if (epoch + 1) % eval_every == 0 or (epoch + 1) == int(train_cfg["epochs"]):
            log_images = True
            metrics = validate(raw_model, val_loader, device, world_size, args.limit_val_batches, rank, "val", log_images, epoch)
            ema_metrics = validate(ema.ema, val_loader, device, world_size, args.limit_val_batches, rank, "val ema", log_images, epoch) if ema else metrics
            score = ema_metrics["psnr"]
            if rank == 0:
                print(
                    f"epoch={epoch + 1} loss={loss_meter.avg:.5f} "
                    f"psnr={metrics['psnr']:.3f} ssim={metrics['ssim']:.4f} "
                    f"ema_psnr={ema_metrics['psnr']:.3f} ema_ssim={ema_metrics['ssim']:.4f}"
                )
                if wandb.run is not None:
                    wandb.log({
                        "val/psnr": metrics["psnr"],
                        "val/ssim": metrics["ssim"],
                        "val_ema/psnr": ema_metrics["psnr"],
                        "val_ema/ssim": ema_metrics["ssim"],
                        "epoch": epoch
                    })

                save_checkpoint(output_dir / "last.pth", model, optimizer, scaler, epoch, best_psnr, cfg, ema)
                if score > best_psnr:
                    best_psnr = score
                    save_checkpoint(output_dir / "best.pth", model, optimizer, scaler, epoch, best_psnr, cfg, ema)
                    if ema is not None:
                        torch.save({"model": ema.ema.state_dict(), "config": cfg, "epoch": epoch, "best_psnr": best_psnr}, output_dir / "best_ema.pth")
                if (epoch + 1) % int(train_cfg["save_every"]) == 0:
                    save_checkpoint(output_dir / f"epoch_{epoch + 1:04d}.pth", model, optimizer, scaler, epoch, best_psnr, cfg, ema)
        else:
            # Optionally just save last checkpoint every epoch without eval
            if rank == 0:
                save_checkpoint(output_dir / "last.pth", model, optimizer, scaler, epoch, best_psnr, cfg, ema)

        if world_size > 1:
            dist.barrier()
    
    if rank == 0 and wandb.run is not None:
        wandb.finish()
        
    cleanup_distributed()


if __name__ == "__main__":
    main()

"""
utils.py – Training utilities: optimizer, scheduler, logging helpers.
"""

import json
import os
from typing import Dict, List

import torch
import torch.nn as nn
from torch.optim import AdamW

# ─── Optimizer ──────────────────────────────────────────────────────────


def build_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    lr_backbone: float = 1e-5,
    weight_decay: float = 1e-4,
) -> AdamW:
    """
    Separate learning rates for backbone vs. transformer/heads.
    Also apply no weight decay to biases and norm layers.
    """
    backbone_params = []
    non_backbone_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # HF Deformable DETR: backbone params have "model.backbone" prefix
        if "backbone" in name:
            backbone_params.append(param)
        else:
            non_backbone_params.append(param)

    # Split weight decay: norm/bias get no decay
    def split_wd(params, lr_, wd):
        no_wd, with_wd = [], []
        for p in params:
            if p.ndim == 1:  # bias / norm weight (1-D)
                no_wd.append(p)
            else:
                with_wd.append(p)
        groups = []
        if with_wd:
            groups.append({"params": with_wd, "lr": lr_, "weight_decay": wd})
        if no_wd:
            groups.append({"params": no_wd, "lr": lr_, "weight_decay": 0.0})
        return groups

    param_groups = split_wd(backbone_params,
                            lr_backbone,
                            weight_decay) + split_wd(non_backbone_params,
                                                     lr,
                                                     weight_decay)
    return AdamW(param_groups)


# ─── LR Scheduler ───────────────────────────────────────────────────────


def build_scheduler(
    optimizer: AdamW,
    epochs: int,
    steps_per_epoch: int,
    warmup_epochs: float = 1.0,
    min_lr_ratio: float = 0.01,
):
    """
    Cosine LR schedule with linear warmup, implemented via LambdaLR.
    Robust for any (epochs, steps_per_epoch) combination.
    """
    import math
    from torch.optim.lr_scheduler import LambdaLR

    total_steps = max(epochs * steps_per_epoch, 1)
    warmup_steps = max(int(warmup_epochs * steps_per_epoch), 1)
    # Clamp so warmup can't exceed total
    warmup_steps = min(warmup_steps, total_steps - 1)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step + 1) / float(warmup_steps)
        # Cosine annealing after warmup
        progress = (current_step - warmup_steps) / \
            max(total_steps - warmup_steps, 1)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine_decay)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


# ─── AMP scaler wrapper ─────────────────────────────────────────────────


class SmoothedValue:
    """Track a series of values and provide access to smoothed means."""

    def __init__(self, window_size: int = 20):
        from collections import deque

        self.window = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.window.append(value)
        self.total += value * n
        self.count += n

    @property
    def avg(self):
        return sum(self.window) / max(len(self.window), 1)

    @property
    def global_avg(self):
        return self.total / max(self.count, 1)

    def __str__(self):
        return f"{self.avg:.4f}"


class MetricLogger:
    def __init__(self, delimiter: str = "  "):
        self.meters: Dict[str, SmoothedValue] = {}
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if k not in self.meters:
                self.meters[k] = SmoothedValue()
            self.meters[k].update(v)

    def __str__(self):
        parts = []
        for k, v in self.meters.items():
            parts.append(f"{k}: {v}")
        return self.delimiter.join(parts)

    def global_avg(self) -> Dict[str, float]:
        return {k: v.global_avg for k, v in self.meters.items()}


# ─── Checkpoint helpers ─────────────────────────────────────────────────


def save_checkpoint(
    output_dir: str,
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    best_map: float,
    metrics: dict,
    name: str = "last.pth",
):
    os.makedirs(output_dir, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_map": best_map,
        "metrics": metrics,
    }
    torch.save(state, os.path.join(output_dir, name))


def load_checkpoint(
    ckpt_path: str,
    model: nn.Module,
    optimizer=None,
    scheduler=None,
    device: torch.device = torch.device("cpu"),
    strict: bool = True,
):
    state = torch.load(ckpt_path, map_location=device)
    missing, unexpected = model.load_state_dict(
        state.get("model", state), strict=strict
    )
    if missing:
        print(f"[WARN] Missing keys: {missing[:3]}...")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected[:3]}...")
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state and state["scheduler"]:
        scheduler.load_state_dict(state["scheduler"])
    return state.get(
        "epoch", 0), state.get(
        "best_map", 0.0), state.get(
            "metrics", {})


# ─── JSON log ───────────────────────────────────────────────────────────


class JsonLogger:
    def __init__(self, path: str):
        self.path = path
        self.records: List[dict] = []
        if os.path.exists(path):
            with open(path) as f:
                self.records = json.load(f)

    def log(self, record: dict):
        self.records.append(record)
        with open(self.path, "w") as f:
            json.dump(self.records, f, indent=2)

"""
data.py – Dataset and augmentation pipeline.

Dataset image size statistics (from 30,062 train images):
  Longer side:  min=25  P25=72   median=104  P75=158  max=876
  Shorter side: min=12  P25=33   median=47   P75=71   max=469
  Aspect ratio: typically ~2.2 (wide landscape strips)

Resize strategy:
  - Resize by LONGER side (not shorter side) to MAX_LONGER=320px.
  - Multi-scale jitter samples longer-side target from TRAIN_SCALES.
  - Skip RandomSizeCrop (images are already tiny; crops kill digits).

Augmentations (train):
  - RandomHorizontalFlip
  - ColorJitter
  - GaussianBlur
  - ResizeLongerSide (multi-scale jitter on longer side)
  - Normalize

Augmentations (val/test):
  - ResizeLongerSide (fixed to MAX_LONGER)
  - Normalize
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as T

# ─── helpers ─────────────────────────────────────────────────────────────────


def _clip_boxes(boxes: torch.Tensor, w: int, h: int) -> torch.Tensor:
    """Clip [x,y,x2,y2] boxes to image boundary."""
    boxes[:, 0].clamp_(0, w)
    boxes[:, 1].clamp_(0, h)
    boxes[:, 2].clamp_(0, w)
    boxes[:, 3].clamp_(0, h)
    return boxes


def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x, y, w, h = boxes.unbind(-1)
    return torch.stack([x, y, x + w, y + h], dim=-1)


def _xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([x1, y1, x2 - x1, y2 - y1], dim=-1)


def _filter_boxes(
    boxes_xyxy: torch.Tensor, labels: torch.Tensor, min_area: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Remove degenerate boxes (w<=0, h<=0, area<min_area)."""
    w = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
    h = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
    keep = (w > 0) & (h > 0) & (w * h >= min_area)
    return boxes_xyxy[keep], labels[keep]


# ─── augmentation transforms ─────────────────────────────────────────────────


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes, labels):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(
            self,
            img: Image.Image,
            boxes: torch.Tensor,
            labels: torch.Tensor):
        if random.random() < self.p:
            w = img.width
            img = TF.hflip(img)
            if boxes.numel() > 0:
                x1 = w - boxes[:, 2]
                x2 = w - boxes[:, 0]
                boxes = torch.stack([x1, boxes[:, 1], x2, boxes[:, 3]], dim=-1)
        return img, boxes, labels


class ColorJitter:
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1):
        self.transform = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue)

    def __call__(self, img, boxes, labels):
        img = self.transform(img)
        return img, boxes, labels


class RandomGrayscale:
    def __init__(self, p: float = 0.1):
        self.p = p

    def __call__(self, img, boxes, labels):
        if random.random() < self.p:
            img = TF.to_grayscale(img, num_output_channels=3)
        return img, boxes, labels


class GaussianBlur:
    def __init__(self, p: float = 0.2, radius_range=(0.5, 2.0)):
        self.p = p
        self.radius_range = radius_range

    def __call__(self, img, boxes, labels):
        if random.random() < self.p:
            radius = random.uniform(*self.radius_range)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img, boxes, labels


class RandomScaleJitter:
    """Randomly resize the short side to one of the given sizes, keeping aspect ratio."""

    def __init__(self, scales: List[int], max_size: int = 1333):
        self.scales = scales
        self.max_size = max_size

    def __call__(
            self,
            img: Image.Image,
            boxes: torch.Tensor,
            labels: torch.Tensor):
        target_size = random.choice(self.scales)
        w0, h0 = img.width, img.height
        scale = target_size / min(w0, h0)
        new_w = int(round(w0 * scale))
        new_h = int(round(h0 * scale))
        # respect max_size
        if max(new_w, new_h) > self.max_size:
            scale2 = self.max_size / max(new_w, new_h)
            new_w = int(round(new_w * scale2))
            new_h = int(round(new_h * scale2))
        img = img.resize((new_w, new_h), Image.BILINEAR)
        if boxes.numel() > 0:
            sx = new_w / w0
            sy = new_h / h0
            boxes = boxes * torch.tensor([sx, sy, sx, sy])
        return img, boxes, labels


class RandomSizeCrop:
    """Random crop, ensuring at least some GT boxes survive."""

    def __init__(self, min_size: int = 256, max_size: int = 800):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(
            self,
            img: Image.Image,
            boxes: torch.Tensor,
            labels: torch.Tensor):
        w, h = img.width, img.height
        # Choose crop size
        cw = random.randint(min(w, self.min_size), min(w, self.max_size))
        ch = random.randint(min(h, self.min_size), min(h, self.max_size))
        # Choose crop position
        x0 = random.randint(0, w - cw)
        y0 = random.randint(0, h - ch)

        # Crop image
        img_crop = img.crop((x0, y0, x0 + cw, y0 + ch))

        if boxes.numel() == 0:
            return img_crop, boxes, labels

        # Adjust boxes
        boxes_new = boxes.clone()
        boxes_new[:, 0] = (boxes[:, 0] - x0).clamp(0, cw)
        boxes_new[:, 1] = (boxes[:, 1] - y0).clamp(0, ch)
        boxes_new[:, 2] = (boxes[:, 2] - x0).clamp(0, cw)
        boxes_new[:, 3] = (boxes[:, 3] - y0).clamp(0, ch)

        # Keep boxes with sufficient overlap
        areas_before = (boxes[:, 2] - boxes[:, 0]) * \
            (boxes[:, 3] - boxes[:, 1])
        areas_after = (boxes_new[:, 2] - boxes_new[:, 0]) * (
            boxes_new[:, 3] - boxes_new[:, 1]
        )
        keep = (areas_after / (areas_before + 1e-6)) > 0.3

        if keep.sum() == 0:
            # Crop killed all boxes – revert
            return img, boxes, labels

        return img_crop, boxes_new[keep], labels[keep]


class Resize:
    """Resize so the short side = target_size, long side ≤ max_size."""

    def __init__(self, target_size: int = 800, max_size: int = 1333):
        self.target_size = target_size
        self.max_size = max_size

    def __call__(
            self,
            img: Image.Image,
            boxes: torch.Tensor,
            labels: torch.Tensor):
        w0, h0 = img.width, img.height
        scale = self.target_size / min(w0, h0)
        new_w = int(round(w0 * scale))
        new_h = int(round(h0 * scale))
        if max(new_w, new_h) > self.max_size:
            scale2 = self.max_size / max(new_w, new_h)
            new_w = int(round(new_w * scale2))
            new_h = int(round(new_h * scale2))
        img = img.resize((new_w, new_h), Image.BILINEAR)
        if boxes.numel() > 0:
            sx = new_w / w0
            sy = new_h / h0
            boxes = boxes * torch.tensor([sx, sy, sx, sy])
        return img, boxes, labels


class ToTensorNormalize:
    """Convert PIL → float tensor, normalize with ImageNet stats."""

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __call__(
            self,
            img: Image.Image,
            boxes: torch.Tensor,
            labels: torch.Tensor):
        img = TF.to_tensor(img)  # [0,1]
        img = TF.normalize(img, self.MEAN, self.STD)
        return img, boxes, labels


# ─── Dataset-tuned constants ─────────────────────────────────────────────────
# Based on full dataset analysis (30,062 images):
#   Longer-side  median=104  P75=158  P95≈320  max=876
# We resize by longer side so landscape strips scale predictably.

# Training multi-scale targets (longer side)
TRAIN_SCALES = [192, 224, 256, 288, 320]  # sample uniformly each iteration

# Hard cap on longer side (prevents huge outliers from exploding memory)
MAX_LONGER = 320

# Validation / test fixed size
VAL_LONGER = 320


# ─── ResizeLongerSide ────────────────────────────────────────────────────────


class ResizeLongerSide:
    """
    Resize image so its LONGER side equals `target_size`,
    preserving aspect ratio.

    Unlike short-side resize (which can blow up wide strips far beyond target),
    this keeps the actual spatial footprint bounded by target_size².
    """

    def __init__(self, target_size: int):
        self.target_size = target_size

    def __call__(
            self,
            img: Image.Image,
            boxes: torch.Tensor,
            labels: torch.Tensor):
        w0, h0 = img.width, img.height
        longer = max(w0, h0)
        if longer == 0:
            return img, boxes, labels
        scale = self.target_size / longer
        new_w = max(1, int(round(w0 * scale)))
        new_h = max(1, int(round(h0 * scale)))
        img = img.resize((new_w, new_h), Image.BILINEAR)
        if boxes.numel() > 0:
            boxes = boxes * torch.tensor([scale, scale, scale, scale])
        return img, boxes, labels


class RandomResizeLongerSide:
    """
    Multi-scale jitter: randomly pick a longer-side target from `scales`
    on every call (per image), then resize.
    """

    def __init__(self, scales: List[int]):
        self.scales = scales

    def __call__(
            self,
            img: Image.Image,
            boxes: torch.Tensor,
            labels: torch.Tensor):
        target = random.choice(self.scales)
        w0, h0 = img.width, img.height
        longer = max(w0, h0)
        if longer == 0:
            return img, boxes, labels
        scale = target / longer
        new_w = max(1, int(round(w0 * scale)))
        new_h = max(1, int(round(h0 * scale)))
        img = img.resize((new_w, new_h), Image.BILINEAR)
        if boxes.numel() > 0:
            boxes = boxes * torch.tensor([scale, scale, scale, scale])
        return img, boxes, labels


# ─── public transform constructors ────────────────────────────────────────


def build_train_transforms(
    scales: Optional[List[int]] = None,
    max_longer: int = MAX_LONGER,
) -> Compose:
    """
    Train augmentation pipeline tailored to tiny digit-strip images.
    Scales refer to the LONGER side (not shorter), matching dataset layout.
    """
    if scales is None:
        scales = TRAIN_SCALES
    return Compose(
        [
            ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            RandomGrayscale(p=0.1),
            GaussianBlur(p=0.2, radius_range=(0.3, 1.0)),
            RandomResizeLongerSide(scales=scales),  # per-image random scale
            ToTensorNormalize(),
        ]
    )


def build_val_transforms(val_longer: int = VAL_LONGER) -> Compose:
    """Validation/test pipeline: resize longer side to val_longer, normalize."""
    return Compose(
        [
            ResizeLongerSide(target_size=val_longer),
            ToTensorNormalize(),
        ]
    )


# ─── Dataset ─────────────────────────────────────────────────────────────────


class DigitDetectionDataset(Dataset):
    """
    COCO-format digit detection dataset.

    category_id 1..10 → model label 0..9
    """

    def __init__(self, img_dir: str, ann_file: str, transforms: Compose):
        self.img_dir = Path(img_dir)
        self.transforms = transforms

        with open(ann_file) as f:
            data = json.load(f)

        self.images: List[Dict] = data["images"]
        self.categories: List[Dict] = data["categories"]

        # Build per-image annotation index
        self._ann_index: Dict[int, List[Dict]] = {}
        for ann in data["annotations"]:
            iid = ann["image_id"]
            self._ann_index.setdefault(iid, []).append(ann)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_info = self.images[idx]
        image_id = img_info["id"]

        # Load image
        img_path = self.img_dir / img_info["file_name"]
        img = Image.open(img_path).convert("RGB")

        anns = self._ann_index.get(image_id, [])

        if anns:
            boxes_xywh = torch.tensor([a["bbox"]
                                      for a in anns], dtype=torch.float32)
            labels = torch.tensor(
                [a["category_id"] - 1 for a in anns],  # 1-indexed → 0-indexed
                dtype=torch.long,
            )
            boxes_xyxy = _xywh_to_xyxy(boxes_xywh)
        else:
            boxes_xyxy = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)

        img, boxes_xyxy, labels = self.transforms(img, boxes_xyxy, labels)

        # Convert back to [cx, cy, w, h] normalized by image size (DETR format)
        h, w = img.shape[-2:]
        boxes_xywh = _xyxy_to_xywh(boxes_xyxy)
        # Normalize to [0, 1]
        boxes_norm = boxes_xywh / \
            torch.tensor([w, h, w, h], dtype=torch.float32)
        # To cx,cy,w,h
        cx = boxes_norm[:, 0] + boxes_norm[:, 2] / 2
        cy = boxes_norm[:, 1] + boxes_norm[:, 3] / 2
        boxes_cxcywh = torch.stack(
            [cx, cy, boxes_norm[:, 2], boxes_norm[:, 3]], dim=-1)

        target = {
            "image_id": image_id,
            "labels": labels,
            "boxes": boxes_cxcywh,  # normalized cx,cy,w,h
            "orig_size": torch.tensor([img_info["height"], img_info["width"]]),
            "size": torch.tensor([h, w]),
        }

        return img, target


class TestDataset(Dataset):
    """Test-only dataset (no annotations)."""

    def __init__(self, img_dir: str, transforms: Compose):
        self.img_dir = Path(img_dir)
        self.transforms = transforms
        self.image_files = (
            sorted(self.img_dir.glob("*.png"))
            + sorted(self.img_dir.glob("*.jpg"))
            + sorted(self.img_dir.glob("*.jpeg"))
        )
        # De-duplicate and sort by numeric stem
        seen = set()
        unique = []
        for p in self.image_files:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        self.image_files = sorted(unique, key=lambda p: int(p.stem))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        p = self.image_files[idx]
        img = Image.open(p).convert("RGB")
        w0, h0 = img.width, img.height
        dummy_boxes = torch.zeros((0, 4), dtype=torch.float32)
        dummy_labels = torch.zeros((0,), dtype=torch.long)
        img, _, _ = self.transforms(img, dummy_boxes, dummy_labels)
        return img, int(p.stem), w0, h0


# ─── Collate ─────────────────────────────────────────────────────────────────


def collate_fn(batch):
    imgs, targets = zip(*batch)
    return list(imgs), list(targets)


def test_collate_fn(batch):
    imgs, image_ids, w0s, h0s = zip(*batch)
    return list(imgs), list(image_ids), list(w0s), list(h0s)

"""
postprocess.py – Post-process Deformable DETR raw outputs into COCO detections.

Steps:
1. Sigmoid scores → top-k per class.
2. cxcywh (normalized) → [x, y, w, h] in pixels.
3. Score threshold.
4. Class-wise soft-NMS / hard-NMS.
5. Global top-k.

Model label 0..9 → category_id 1..10.
"""

import torch
import torchvision
from typing import List, Dict

# ─── box utilities ──────────────────────────────────────────────────────


def _cx_cy_wh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (cx, cy, w, h) in [0,1] to (x1, y1, x2, y2) normalized."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack(
        [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def _xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([x1, y1, x2 - x1, y2 - y1], dim=-1)


# ─── NMS ─────────────────────────────────────────────────────────────────────


def batched_nms(
    boxes_xyxy: torch.Tensor,  # (N, 4) pixel coords
    scores: torch.Tensor,  # (N,)
    labels: torch.Tensor,  # (N,) class indices
    iou_threshold: float = 0.5,
) -> torch.Tensor:
    """Torchvision batched NMS; returns kept indices."""
    if boxes_xyxy.numel() == 0:
        return torch.zeros(0, dtype=torch.long)
    return torchvision.ops.batched_nms(
        boxes_xyxy, scores, labels, iou_threshold)


# ─── main postprocessor ─────────────────────────────────────────────────


class DeformableDetrPostProcessor:
    """
    Converts model logits + pred_boxes into COCO-format detections.

    Args:
        score_threshold: Minimum sigmoid score to keep a detection.
        nms_iou_threshold: IoU threshold for class-wise NMS.
        max_detections: Keep at most this many detections per image.
        num_classes: Number of classes (10 for digits).
    """

    def __init__(
        self,
        score_threshold: float = 0.3,
        nms_iou_threshold: float = 0.5,
        max_detections: int = 300,
        num_classes: int = 10,
    ):
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections = max_detections
        self.num_classes = num_classes

    @torch.no_grad()
    def __call__(
        self,
        logits: torch.Tensor,  # (B, Q, C)
        pred_boxes: torch.Tensor,  # (B, Q, 4)  cx,cy,w,h normalized
        orig_sizes: torch.Tensor,  # (B, 2)  [H, W]
        image_ids: List[int],
    ) -> List[Dict]:
        """
        Returns a list of COCO-format dicts:
          {image_id, bbox:[x,y,w,h], score, category_id}
        """
        B = logits.shape[0]
        scores_all = logits.sigmoid()  # (B, Q, C)

        results = []
        for b in range(B):
            h, w = orig_sizes[b].tolist()
            scores_b = scores_all[b]  # (Q, C)
            boxes_b = pred_boxes[b]  # (Q, 4)  normalized cxcywh

            # Convert to absolute pixel coords (xyxy)
            boxes_xyxy = _cx_cy_wh_to_xyxy(boxes_b)  # normalized xyxy
            boxes_xyxy = boxes_xyxy * torch.tensor(
                [w, h, w, h], dtype=torch.float32, device=boxes_xyxy.device
            )
            # Clamp to image
            boxes_xyxy[:, 0].clamp_(0, w)
            boxes_xyxy[:, 1].clamp_(0, h)
            boxes_xyxy[:, 2].clamp_(0, w)
            boxes_xyxy[:, 3].clamp_(0, h)

            # Expand: one detection per query × class
            # scores_b: (Q, C), boxes: (Q, 4) → replicate queries for each
            # class
            Q, C = scores_b.shape
            # For each class, gather top predictions
            all_scores = []
            all_labels = []
            all_boxes = []
            for c in range(C):
                sc = scores_b[:, c]  # (Q,)
                mask = sc >= self.score_threshold
                if mask.sum() == 0:
                    continue
                all_scores.append(sc[mask])
                all_labels.append(
                    torch.full(
                        (mask.sum(),), c, dtype=torch.long, device=sc.device))
                all_boxes.append(boxes_xyxy[mask])

            if not all_scores:
                # No detection for this image
                continue

            all_scores = torch.cat(all_scores)
            all_labels = torch.cat(all_labels)
            all_boxes = torch.cat(all_boxes, dim=0)

            # Class-wise NMS
            keep = batched_nms(
                all_boxes, all_scores, all_labels, self.nms_iou_threshold
            )
            all_scores = all_scores[keep]
            all_labels = all_labels[keep]
            all_boxes = all_boxes[keep]

            # Global top-k
            if all_scores.shape[0] > self.max_detections:
                topk = torch.topk(all_scores, self.max_detections).indices
                all_scores = all_scores[topk]
                all_labels = all_labels[topk]
                all_boxes = all_boxes[topk]

            # Convert xyxy → xywh (COCO format)
            boxes_xywh = _xyxy_to_xywh(all_boxes)

            image_id = image_ids[b]
            for i in range(all_scores.shape[0]):
                bx, by, bw, bh = boxes_xywh[i].tolist()
                if bw <= 0 or bh <= 0:
                    continue
                results.append(
                    {
                        "image_id": image_id,
                        "bbox": [
                            round(bx, 2),
                            round(by, 2),
                            round(bw, 2),
                            round(bh, 2),
                        ],
                        "score": round(all_scores[i].item(), 4),
                        # 0..9 → 1..10
                        "category_id": int(all_labels[i].item()) + 1,
                    }
                )

        return results

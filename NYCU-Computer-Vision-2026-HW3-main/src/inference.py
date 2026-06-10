import os
import json
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm
from pycocotools import mask as mask_util
from ensemble_boxes import weighted_boxes_fusion

from dataset import CellDataset, get_test_transforms
from model import get_model_instance_segmentation

MAX_SIZE = 800  # Must match dataset.py


def encode_mask_to_rle(mask):
    """Encode binary mask (H, W) uint8 to COCO RLE string."""
    rle = mask_util.encode(np.asfortranarray(mask))
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def _compute_padding(orig_h, orig_w, tta_scale, max_size=MAX_SIZE):
    """Return (new_h, new_w, pad_top, pad_left) for a given TTA scale factor.

    Mirrors exactly what Albumentations LongestMaxSize + PadIfNeeded does:
      1. LongestMaxSize scales so longest side == max_size (capped, never upscale unless TTA > 1)
      2. PadIfNeeded center-pads to the next multiple of 32.
    """
    effective_max = int(round(max_size * tta_scale))
    base_scale = effective_max / float(max(orig_h, orig_w))
    new_h = int(round(orig_h * base_scale))
    new_w = int(round(orig_w * base_scale))

    pad_h = (32 - new_h % 32) % 32
    pad_w = (32 - new_w % 32) % 32
    pad_top = pad_h // 2
    pad_left = pad_w // 2
    return new_h, new_w, pad_top, pad_left


def _predict_single_scale(model, image_orig, orig_h, orig_w, tta_scale, device):
    """Run inference for one TTA scale. Returns boxes/scores/labels/masks in ORIG coords."""
    new_h, new_w, pad_top, pad_left = _compute_padding(orig_h, orig_w, tta_scale)

    # Resize image
    img_resized = torch.nn.functional.interpolate(
        image_orig.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False
    ).squeeze(0)

    # Center-pad to multiple of 32 (same as Albumentations PadIfNeeded)
    pad_bottom = (32 - new_h % 32) % 32 - pad_top
    pad_right  = (32 - new_w % 32) % 32 - pad_left
    img_padded = torch.nn.functional.pad(
        img_resized, (pad_left, pad_right, pad_top, pad_bottom), value=0.0
    )

    preds = model([img_padded.to(device)])[0]
    boxes  = preds['boxes']
    scores = preds['scores']
    labels = preds['labels']
    masks  = preds['masks']  # (N, 1, H_pad, W_pad)

    if boxes.shape[0] == 0:
        return (np.zeros((0, 4)), np.zeros((0,)),
                np.zeros((0,), dtype=np.int64), np.zeros((0, orig_h, orig_w), dtype=np.float32))

    # ── Reverse padding ──────────────────────────────────────────────────────
    boxes = boxes.clone()
    boxes[:, 0] -= pad_left
    boxes[:, 1] -= pad_top
    boxes[:, 2] -= pad_left
    boxes[:, 3] -= pad_top
    boxes[:, 0] = torch.clamp(boxes[:, 0], min=0, max=new_w)
    boxes[:, 1] = torch.clamp(boxes[:, 1], min=0, max=new_h)
    boxes[:, 2] = torch.clamp(boxes[:, 2], min=0, max=new_w)
    boxes[:, 3] = torch.clamp(boxes[:, 3], min=0, max=new_h)

    # ── Reverse scaling ───────────────────────────────────────────────────────
    scale_x = float(new_w) / orig_w
    scale_y = float(new_h) / orig_h
    boxes[:, 0] /= scale_x
    boxes[:, 1] /= scale_y
    boxes[:, 2] /= scale_x
    boxes[:, 3] /= scale_y

    # ── Reverse mask padding + scale masks to orig size ──────────────────────
    masks = masks[:, :, pad_top : pad_top + new_h, pad_left : pad_left + new_w]
    if masks.shape[0] > 0:
        masks = torch.nn.functional.interpolate(
            masks, size=(orig_h, orig_w), mode='bilinear', align_corners=False
        )

    return (
        boxes.cpu().numpy(),
        scores.cpu().numpy(),
        labels.cpu().numpy().astype(np.int64),
        masks.squeeze(1).cpu().numpy(),
    )


def _split_connected_components(binary_mask, score, label):
    """Split a binary mask into connected components.
    Returns list of (mask, score, label) for each component."""
    num_labels, comp_map = cv2.connectedComponents(binary_mask.astype(np.uint8))
    if num_labels <= 2:  # 0 = background, 1 = single component
        return [(binary_mask, score, label)]

    results = []
    total_area = binary_mask.sum()
    for comp_id in range(1, num_labels):
        comp_mask = (comp_map == comp_id).astype(np.uint8)
        comp_area = comp_mask.sum()
        if comp_area < 10:  # skip tiny noise blobs
            continue
        # Scale score proportionally to relative area
        comp_score = score * (comp_area / total_area)
        results.append((comp_mask, float(comp_score), label))
    return results if results else [(binary_mask, score, label)]


@torch.no_grad()
def generate_submission(model, data_loader, device, test_ids_path, output_path,
                        tta_scales=(0.8, 1.0, 1.2),
                        wbf_iou_thr=0.55, wbf_skip_box_thr=0.001,
                        score_threshold=0.001, max_preds=300):
    model.eval()

    with open(test_ids_path, 'r') as f:
        name_to_id_mapping = json.load(f)

    name_to_id   = {item['file_name']: item['id']   for item in name_to_id_mapping}
    name_to_info = {item['file_name']: item          for item in name_to_id_mapping}

    results = []

    progress_bar = tqdm(data_loader, desc="Inference (TTA+WBF)")
    for images, filenames in progress_bar:
        image    = images[0]   # keep on CPU; we push per-scale later
        filename = filenames[0]

        image_id = name_to_id.get(filename)
        info     = name_to_info.get(filename)
        if image_id is None or info is None:
            print(f"Warning: {filename} not found in {test_ids_path}")
            continue

        orig_h, orig_w = info['height'], info['width']

        # ── Collect predictions from all TTA scales ───────────────────────────
        all_boxes_list  = []   # list of (N, 4) normalised [0,1]
        all_scores_list = []   # list of (N,)
        all_labels_list = []   # list of (N,) int
        all_masks_raw   = []   # list of (N, orig_h, orig_w) float [0,1]

        for tta_scale in tta_scales:
            boxes, scores, labels, masks = _predict_single_scale(
                model, image, orig_h, orig_w, tta_scale, device
            )
            if boxes.shape[0] == 0:
                all_boxes_list.append(np.zeros((0, 4)))
                all_scores_list.append(np.zeros((0,)))
                all_labels_list.append(np.zeros((0,), dtype=np.int64))
                all_masks_raw.append(np.zeros((0, orig_h, orig_w), dtype=np.float32))
                continue

            # Normalise boxes to [0, 1] for WBF
            norm = np.array([orig_w, orig_h, orig_w, orig_h], dtype=np.float32)
            boxes_norm = np.clip(boxes / norm, 0.0, 1.0)

            all_boxes_list.append(boxes_norm)
            all_scores_list.append(scores)
            all_labels_list.append(labels)
            all_masks_raw.append(masks)

        # ── WBF per category ──────────────────────────────────────────────────
        # WBF needs all predictions flattened; we run it once globally then
        # assign the best-matching raw mask to each fused box.
        all_boxes_flat  = np.concatenate(all_boxes_list,  axis=0)
        all_scores_flat = np.concatenate(all_scores_list, axis=0)
        all_labels_flat = np.concatenate(all_labels_list, axis=0)
        all_masks_flat  = np.concatenate(all_masks_raw,   axis=0)  # (TotalN, H, W)

        if all_boxes_flat.shape[0] == 0:
            continue

        # WBF: run per-category so labels stay consistent
        fused_boxes_all  = []
        fused_scores_all = []
        fused_labels_all = []
        fused_masks_all  = []

        for cat in np.unique(all_labels_flat):
            cat_mask = all_labels_flat == cat
            cat_boxes  = all_boxes_flat[cat_mask]
            cat_scores = all_scores_flat[cat_mask]
            cat_masks  = all_masks_flat[cat_mask]

            fb, fs, _ = weighted_boxes_fusion(
                [cat_boxes],
                [cat_scores],
                [np.zeros(len(cat_scores))],  # single "model" index
                iou_thr=wbf_iou_thr,
                skip_box_thr=wbf_skip_box_thr,
                conf_type='avg',
            )
            if len(fb) == 0:
                continue

            # Match each fused box to the raw mask with highest IoU
            for fbox, fscore in zip(fb, fs):
                # IoU between fused box and all raw boxes of this category
                def box_iou_1toN(b1, bN):
                    xi1 = np.maximum(b1[0], bN[:, 0])
                    yi1 = np.maximum(b1[1], bN[:, 1])
                    xi2 = np.minimum(b1[2], bN[:, 2])
                    yi2 = np.minimum(b1[3], bN[:, 3])
                    inter = np.maximum(0, xi2 - xi1) * np.maximum(0, yi2 - yi1)
                    area1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
                    areaN = (bN[:, 2]-bN[:, 0]) * (bN[:, 3]-bN[:, 1])
                    union = area1 + areaN - inter
                    return inter / np.maximum(union, 1e-6)

                ious = box_iou_1toN(fbox, cat_boxes)
                best_idx = int(np.argmax(ious))
                fused_masks_all.append(cat_masks[best_idx])
                fused_boxes_all.append(fbox)
                fused_scores_all.append(fscore)
                fused_labels_all.append(int(cat))

        if len(fused_boxes_all) == 0:
            continue

        fused_boxes  = np.array(fused_boxes_all)   # (M, 4) normalised
        fused_scores = np.array(fused_scores_all)
        fused_labels = np.array(fused_labels_all, dtype=np.int64)
        fused_masks  = np.array(fused_masks_all)   # (M, orig_h, orig_w)

        # Denormalise boxes back to pixel coords
        norm = np.array([orig_w, orig_h, orig_w, orig_h], dtype=np.float32)
        fused_boxes = fused_boxes * norm

        # ── Sort by score, keep top max_preds ─────────────────────────────────
        order = np.argsort(fused_scores)[::-1][:max_preds]
        fused_boxes  = fused_boxes[order]
        fused_scores = fused_scores[order]
        fused_labels = fused_labels[order]
        fused_masks  = fused_masks[order]

        # ── Emit predictions + Connected Components split ─────────────────────
        for i in range(len(fused_scores)):
            if fused_scores[i] < score_threshold:
                continue

            binary_mask = (fused_masks[i] > 0.5).astype(np.uint8)
            if binary_mask.sum() == 0:
                continue

            components = _split_connected_components(
                binary_mask, fused_scores[i], fused_labels[i]
            )

            for comp_mask, comp_score, comp_label in components:
                rows = np.any(comp_mask, axis=1)
                cols = np.any(comp_mask, axis=0)
                if not rows.any():
                    continue
                y1, y2 = np.where(rows)[0][[0, -1]]
                x1, x2 = np.where(cols)[0][[0, -1]]
                w_box = float(x2 - x1 + 1)
                h_box = float(y2 - y1 + 1)

                results.append({
                    "image_id":    image_id,
                    "bbox":        [float(x1), float(y1), w_box, h_box],
                    "score":       comp_score,
                    "category_id": int(comp_label),
                    "segmentation": encode_mask_to_rle(comp_mask),
                })

    print(f"Saving {len(results)} predictions to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(results, f)


def main():
    data_dir         = "test_release"
    test_ids_path    = "test_image_name_to_ids.json"
    checkpoint_path  = "/708HDD/pkc776/checkpoints/run_20260512_014648/best_model.pth"
    output_path      = "test-results.json"
    num_classes      = 5

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # Raw test images without albumentations — we handle all transforms manually per TTA scale
    test_dataset = CellDataset(data_dir, split="test", transforms=None)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    model = get_model_instance_segmentation(num_classes)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found. Using untrained weights.")

    model.to(device)

    generate_submission(model, test_loader, device, test_ids_path, output_path)
    print("Done. Compress test-results.json into a zip file for submission.")


if __name__ == "__main__":
    main()

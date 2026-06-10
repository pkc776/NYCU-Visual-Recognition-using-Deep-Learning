"""
evaluate.py – COCO mAP evaluation using pycocotools.

Usage:
    from deformable_detr_hw2.evaluate import CocoEvaluator
    evaluator = CocoEvaluator(ann_file="nycu-hw2-data/valid.json")
    evaluator.update(predictions)   # list of COCO dicts
    metrics = evaluator.summarize()
    # → {"mAP": ..., "AP50": ..., "AP75": ..., ...}
"""

import json
import tempfile
import os
from typing import List, Dict

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class CocoEvaluator:
    """
    Wraps pycocotools COCOeval for computing mAP on the validation set.

    Args:
        ann_file: Path to the COCO-format ground-truth JSON.
    """

    def __init__(self, ann_file: str):
        self.coco_gt = COCO(ann_file)
        self._predictions: List[Dict] = []

    def reset(self):
        self._predictions = []

    def update(self, predictions: List[Dict]):
        """
        Accumulate predictions.

        Each prediction dict must have:
            image_id, bbox ([x,y,w,h] pixel), score, category_id
        """
        self._predictions.extend(predictions)

    def summarize(self) -> Dict[str, float]:
        """
        Run COCOeval and return dict with mAP metrics.
        Returns zeros if no predictions were added.
        """
        if not self._predictions:
            return {
                "mAP": 0.0,
                "AP50": 0.0,
                "AP75": 0.0,
                "AP_s": 0.0,
                "AP_m": 0.0,
                "AP_l": 0.0,
                "AR_1": 0.0,
                "AR_10": 0.0,
                "AR_100": 0.0,
            }

        # Write predictions to a temp file so COCO can load them
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self._predictions, f)
            tmp_path = f.name

        try:
            coco_dt = self.coco_gt.loadRes(tmp_path)
            coco_eval = COCOeval(self.coco_gt, coco_dt, "bbox")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            stats = coco_eval.stats  # ndarray of 12 values

            return {
                "mAP": float(stats[0]),  # AP @ IoU=0.50:0.95
                "AP50": float(stats[1]),  # AP @ IoU=0.50
                "AP75": float(stats[2]),  # AP @ IoU=0.75
                "AP_s": float(stats[3]),  # AP small
                "AP_m": float(stats[4]),  # AP medium
                "AP_l": float(stats[5]),  # AP large
                "AR_1": float(stats[6]),  # AR max=1
                "AR_10": float(stats[7]),  # AR max=10
                "AR_100": float(stats[8]),  # AR max=100
            }
        finally:
            os.unlink(tmp_path)

import torch
import torch.nn as nn
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.modeling import build_model
from detectron2.structures import Instances, Boxes, BitMasks
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool
from detectron2.layers import ShapeSpec
import timm

class TimmBackbone(Backbone):
    def __init__(self, name, out_indices):
        super().__init__()
        # features_only=True returns intermediate features for FPN
        self.model = timm.create_model(name, features_only=True, out_indices=out_indices, pretrained=True)
        self._out_features = [f"res{i+2}" for i in range(len(out_indices))]
        
        feature_info = self.model.feature_info.info
        self._out_feature_channels = {f"res{i+2}": info['num_chs'] for i, info in enumerate(feature_info)}
        self._out_feature_strides = {f"res{i+2}": info['reduction'] for i, info in enumerate(feature_info)}

    def forward(self, x):
        features = self.model(x)
        return {name: f for name, f in zip(self._out_features, features)}

    def output_shape(self):
        return {
            name: ShapeSpec(channels=self._out_feature_channels[name], stride=self._out_feature_strides[name])
            for name in self._out_features
        }

@BACKBONE_REGISTRY.register()
def build_convnextv2_fpn_backbone(cfg, input_shape: ShapeSpec):
    # For ConvNeXt-v2-base, we extract from 4 stages (indices 0, 1, 2, 3)
    bottom_up = TimmBackbone("convnextv2_base", out_indices=(0, 1, 2, 3))
    in_features = bottom_up._out_features
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone

class CascadeMaskRCNNConvNeXtV2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        cfg = get_cfg()
        # Use Cascade Mask R-CNN meta architecture from Detectron2
        cfg.merge_from_file(model_zoo.get_config_file("Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml"))
        
        # Replace backbone with ConvNeXt-V2-Base + FPN
        cfg.MODEL.BACKBONE.NAME = "build_convnextv2_fpn_backbone"
        cfg.MODEL.WEIGHTS = "" # Start from fresh head/FPN, timm provides backbone pretrained weights
        
        # In detectron2, classes include only foreground classes!
        # Torchvision num_classes=5 (including background). We pass 4 to D2.
        self.d2_num_classes = num_classes - 1
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.d2_num_classes
        
        # The user's dataset yields image tensors in [0, 1] scale in RGB.
        # We adjust detectron2's preprocessing to match ImageNet standard normalization on [0, 1].
        cfg.MODEL.PIXEL_MEAN = [0.485, 0.456, 0.406]
        cfg.MODEL.PIXEL_STD = [0.229, 0.224, 0.225]
        cfg.INPUT.FORMAT = "RGB" 
        
        self.model = build_model(cfg)

    def forward(self, images, targets=None):
        """
        Expects images as a list of float tensors in [0, 1], format (C, H, W)
        Expects targets as a list of dicts with 'boxes', 'labels', 'masks' (torchvision format)
        """
        batched_inputs = []
        for i, img in enumerate(images):
            # The detectron2 model internally applies preprocessing: (x - PIXEL_MEAN) / PIXEL_STD
            d2_input = {"image": img}
            
            if targets is not None:
                target = targets[i]
                instances = Instances(img.shape[1:])
                instances.gt_boxes = Boxes(target["boxes"])
                # torchvision labels are 1-indexed (0 is background). Detectron2 uses 0-indexed for foreground.
                instances.gt_classes = target["labels"] - 1
                instances.gt_masks = BitMasks(target["masks"])
                d2_input["instances"] = instances
            
            batched_inputs.append(d2_input)

        if self.training:
            # Returns a dictionary of losses
            from detectron2.utils.events import EventStorage
            with EventStorage():
                return self.model(batched_inputs)
        else:
            # In eval mode, detectron2 model requires batched_inputs but returns a list of dicts
            self.model.eval()
            with torch.no_grad():
                d2_preds = self.model(batched_inputs)
            self.model.train() # restore training mode since we manage modes via nn.Module.train() wrapper
            
            torchvision_preds = []
            for p in d2_preds:
                inst = p["instances"]
                # Convert back to torchvision format
                pred = {
                    "boxes": inst.pred_boxes.tensor,
                    "scores": inst.scores,
                    "labels": inst.pred_classes + 1, # Convert back to 1-indexed
                }
                if inst.has("pred_masks"):
                    # Detectron2 masks are (N, H, W). Torchvision expects (N, 1, H, W)
                    masks = inst.pred_masks.float().unsqueeze(1)
                    pred["masks"] = masks
                torchvision_preds.append(pred)
            return torchvision_preds

def get_model_instance_segmentation(num_classes):
    """
    Returns a Cascade Mask R-CNN model with a ConvNeXt-V2-Base FPN backbone
    configured for the given number of classes.
    """
    return CascadeMaskRCNNConvNeXtV2(num_classes)

def print_model_parameters(model):
    """Prints total number of trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params / 1e6:.2f} M")
    return total_params

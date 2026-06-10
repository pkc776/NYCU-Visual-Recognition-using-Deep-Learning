"""
modeling.py – Deformable DETR model builder.

Uses HuggingFace `transformers.DeformableDetrForObjectDetection`
with a ResNet-50 backbone (pretrained on ImageNet).

num_labels = 10 (digits 0-9)
"""

import torch
import torch.nn as nn
from transformers import (
    DeformableDetrForObjectDetection,
    DeformableDetrConfig,
)

NUM_CLASSES = 10  # digits 0-9 (category_id 1-10 remapped to 0-9)


def build_model(
    num_queries: int = 300,
    num_encoder_layers: int = 6,
    num_decoder_layers: int = 6,
    d_model: int = 256,
    pretrained_backbone: bool = True,
    two_stage: bool = False,
    num_feature_levels: int = 4,
) -> DeformableDetrForObjectDetection:
    """
    Build a Deformable DETR model with ResNet-50 backbone.

    Args:
        num_queries: Number of object queries.
        num_encoder_layers: Transformer encoder depth.
        num_decoder_layers: Transformer decoder depth.
        d_model: Feature dimension (hidden size).
        pretrained_backbone: Use ImageNet pretrained weights for ResNet-50.
        two_stage: Enable two-stage Deformable DETR (region proposals).
        num_feature_levels: Multi-scale feature levels (default 4).

    Returns:
        DeformableDetrForObjectDetection instance.
    """
    backbone_name = "resnet50"

    config = DeformableDetrConfig(
        # Backbone
        backbone=backbone_name,
        use_pretrained_backbone=pretrained_backbone,
        backbone_kwargs=(
            {"out_indices": [1, 2, 3, 4]} if num_feature_levels == 4 else None
        ),
        # Architecture
        d_model=d_model,
        encoder_layers=num_encoder_layers,
        decoder_layers=num_decoder_layers,
        encoder_attention_heads=8,
        decoder_attention_heads=8,
        encoder_ffn_dim=1024,
        decoder_ffn_dim=1024,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        num_queries=num_queries,
        num_feature_levels=num_feature_levels,
        # Two-stage
        two_stage=two_stage,
        # Detection head
        num_labels=NUM_CLASSES,
        # Loss weights
        bbox_cost=5.0,
        giou_cost=2.0,
        class_cost=2.0,
        bbox_loss_coefficient=5.0,
        giou_loss_coefficient=2.0,
        # Misc
        auxiliary_loss=True,
        with_box_refine=True,
    )

    model = DeformableDetrForObjectDetection(config)

    if pretrained_backbone:
        print("[INFO] Loading ImageNet pretrained weights for ResNet-50 backbone ONLY.")
        try:
            import timm

            timm_model = timm.create_model(backbone_name, pretrained=True)
            model.model.backbone.conv_encoder.model.load_state_dict(
                timm_model.state_dict(), strict=False
            )
            print("[INFO] Backbone weights loaded successfully.")
        except Exception as e:
            print(f"[WARN] Failed to load pretrained backbone weights: {e}")
    else:
        print("[INFO] Building model completely from scratch (no pretrained weights).")

    return model


def load_checkpoint(
    model: nn.Module, ckpt_path: str, device: torch.device, strict: bool = True
) -> dict:
    """Load a checkpoint; returns the saved metadata dict."""
    state = torch.load(ckpt_path, map_location=device)
    model_state = state.get("model", state)
    missing, unexpected = model.load_state_dict(model_state, strict=strict)
    if missing:
        print(f"[WARN] Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(
            f"[WARN] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    return state

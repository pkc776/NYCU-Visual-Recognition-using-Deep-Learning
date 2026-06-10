# NYCU Computer Vision 2026 HW4

- Student ID: 112550127
- Name: Pin-Kuan Chiang

## Introduction

This repository contains the key code for HW4 rain/snow image restoration. The task is to train a single PromptIR-based model that restores both rain-degraded and snow-degraded images.

My method uses PromptIR as the base all-in-one restoration framework and modifies the restoration network with NAFNet-style residual blocks, residual prediction, EMA training, restoration-specific losses, and physically valid augmentations. No external data or external pretrained weights are used. The final continued-training stage starts from my own EMA checkpoint trained from scratch on the official HW4 training set.

## Environment Setup

Create and activate a Python environment, then install dependencies:

```bash
conda create -n cvdl_hw4 python=3.10 -y
conda activate cvdl_hw4
pip install -r requirements.txt
```

The code is intended for PyTorch with CUDA GPUs.

## Usage

Place the released dataset under:

```text
release_folder/hw4_realse_dataset/
  train/degraded/
  train/clean/
  test/degraded/
```

Check the dataset split:

```bash
python -m src.promptir_hw4.data --root release_folder/hw4_realse_dataset
```

Train from scratch:

```bash
python -m src.promptir_hw4.train --config configs/promptir_hw4.yaml
```

Multi-GPU training:

```bash
bash scripts/train_ddp.sh
```

Continue training from my own scratch-trained checkpoint:

```bash
python -m src.promptir_hw4.train \
  --config configs/promptir_hw4_ft256.yaml \
  --init-checkpoint outputs/promptir_hw4_4stage_w32_drop01_no_color_jitter/best_ema.pth
```

Run inference 

```bash
python -m src.promptir_hw4.infer \
  --config configs/promptir_hw4.yaml \
  --checkpoint outputs/promptir_hw4_ft256_from_ema/best_ema.pth \
  --tta \
  --output pred.npz
```

Validate the submission file:

```bash
python -m src.promptir_hw4.validate_submission \
  --npz pred.npz \
  --test-dir release_folder/hw4_realse_dataset/test/degraded
```

## Performance Snapshot


![Performance snapshot](image.png)

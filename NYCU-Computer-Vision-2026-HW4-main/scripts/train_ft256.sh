#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/promptir_hw4_ft256.yaml}
INIT_CHECKPOINT=${INIT_CHECKPOINT:-outputs/promptir_hw4_4stage_w32_drop01_no_color_jitter/best_ema.pth}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  -m src.promptir_hw4.train \
  --config "${CONFIG}" \
  --init-checkpoint "${INIT_CHECKPOINT}" \
  "$@"

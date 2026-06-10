#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/promptir_hw4.yaml}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  -m src.promptir_hw4.train \
  --config "${CONFIG}" \
  "$@"

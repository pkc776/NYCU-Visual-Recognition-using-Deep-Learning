#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/promptir_hw4.yaml}
CHECKPOINT=${CHECKPOINT:-outputs/promptir_hw4_ft256_from_ema/best_ema.pth}
OUTPUT=${OUTPUT:-pred.npz}

python -m src.promptir_hw4.infer \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --tta \
  --output "${OUTPUT}" \
  "$@"

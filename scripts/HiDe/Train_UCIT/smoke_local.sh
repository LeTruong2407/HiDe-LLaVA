#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE_DATA_JSON="$(bash "$SCRIPT_DIR/prepare_smoke_data.sh")"

export TRAIN_LAUNCHER=python
export TRAIN_BF16=False
export TRAIN_TF32=False
export TRAIN_PER_DEVICE_BATCH=1
export TRAIN_PER_DEVICE_EVAL_BATCH=1
export TRAIN_GRAD_ACCUM_STEPS=1
export TRAIN_NUM_WORKERS=0
export TRAIN_SAVE_STRATEGY=no
export TRAIN_LOGGING_STEPS=1
export EXTRA_TRAIN_ARGS="--max_steps 1 --no_cuda True"
export UCIT_TASK1_TRAIN_JSON="$SMOKE_DATA_JSON"

echo "Running a 1-step local smoke test for UCIT task 1"
bash "$SCRIPT_DIR/Task1.sh"

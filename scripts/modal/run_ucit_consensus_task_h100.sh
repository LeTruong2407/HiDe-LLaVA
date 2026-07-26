#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
TASK_NUMBER="${TASK_NUMBER:-2}"
PROFILE="${PROFILE:-balanced}"
QUANT_MODE="${QUANT_MODE:-bf16}"
TRAIN_BATCH="${TRAIN_BATCH:-64}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-1024}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-256}"
SAMPLES_PER_FORWARD="${SAMPLES_PER_FORWARD:-8}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"

if ! [[ "$TASK_NUMBER" =~ ^[1-6]$ ]]; then
  echo "TASK_NUMBER must be between 1 and 6." >&2
  exit 2
fi

COMMAND=(
  "$PYTHON_BIN" -m modal run --detach
  modal_imagenet_r_consensus.py
  --action train
  --task-number "$TASK_NUMBER"
  --profile "$PROFILE"
  --quant-mode "$QUANT_MODE"
  --train-batch "$TRAIN_BATCH"
  --grad-accum "$GRAD_ACCUM"
  --model-max-length "$MODEL_MAX_LENGTH"
  --sample-limit "$SAMPLE_LIMIT"
  --samples-per-forward "$SAMPLES_PER_FORWARD"
  --logging-steps "$LOGGING_STEPS"
  --save-strategy "$SAVE_STRATEGY"
  --save-steps "$SAVE_STEPS"
  --save-total-limit "$SAVE_TOTAL_LIMIT"
)

if [ -n "${MAX_STEPS:-}" ]; then
  COMMAND+=(--max-steps "$MAX_STEPS")
fi

if [ -n "${RESUME_FROM_CHECKPOINT:-}" ]; then
  COMMAND+=(--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT")
fi

echo "Launching detached H100 UCIT Task $TASK_NUMBER training:"
printf ' %q' "${COMMAND[@]}"
echo
exec "${COMMAND[@]}"

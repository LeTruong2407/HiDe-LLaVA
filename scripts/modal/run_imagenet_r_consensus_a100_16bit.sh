#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODAL_DETACH="${MODAL_DETACH:-1}"

PROFILE="${PROFILE:-aggressive}"
TRAIN_BATCH="${TRAIN_BATCH:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-1024}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-256}"
SAMPLES_PER_FORWARD="${SAMPLES_PER_FORWARD:-8}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"

if [ "$MODAL_DETACH" = "1" ]; then
  exec "$PYTHON_BIN" -m modal run --detach modal_imagenet_r_consensus.py \
    --action train \
    --profile "$PROFILE" \
    --quant-mode bf16 \
    --train-batch "$TRAIN_BATCH" \
    --grad-accum "$GRAD_ACCUM" \
    --model-max-length "$MODEL_MAX_LENGTH" \
    --sample-limit "$SAMPLE_LIMIT" \
    --samples-per-forward "$SAMPLES_PER_FORWARD" \
    --logging-steps "$LOGGING_STEPS"
else
  exec "$PYTHON_BIN" -m modal run modal_imagenet_r_consensus.py \
    --action train \
    --profile "$PROFILE" \
    --quant-mode bf16 \
    --train-batch "$TRAIN_BATCH" \
    --grad-accum "$GRAD_ACCUM" \
    --model-max-length "$MODEL_MAX_LENGTH" \
    --sample-limit "$SAMPLE_LIMIT" \
    --samples-per-forward "$SAMPLES_PER_FORWARD" \
    --logging-steps "$LOGGING_STEPS"
fi

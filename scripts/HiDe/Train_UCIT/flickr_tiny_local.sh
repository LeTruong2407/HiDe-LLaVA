#!/bin/bash
set -euo pipefail

################## VICUNA ##################
PROMPT_VERSION=v1
MODEL_VERSION="vicuna-7b-v1.5"
################## VICUNA ##################

################## LLaMA-2 ##################
# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="Llama-2-7b-chat-hf"
################## LLaMA-2 ##################

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

if [ ! -f "$FLICKR_TINY_TRAIN_JSON" ]; then
  echo "Missing tiny Flickr train JSON: $FLICKR_TINY_TRAIN_JSON"
  echo "Run:"
  echo "  bash scripts/download/download_flickr30k_tiny.sh 32"
  echo "  bash scripts/download/make_tiny_flickr30k_split.sh 32"
  exit 1
fi

export TRAIN_LAUNCHER=python
export TRAIN_BF16=False
export TRAIN_TF32=False
export TRAIN_PER_DEVICE_BATCH="${TRAIN_PER_DEVICE_BATCH:-1}"
export TRAIN_PER_DEVICE_EVAL_BATCH="${TRAIN_PER_DEVICE_EVAL_BATCH:-1}"
export TRAIN_GRAD_ACCUM_STEPS="${TRAIN_GRAD_ACCUM_STEPS:-1}"
export TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-0}"
export TRAIN_SAVE_STRATEGY="${TRAIN_SAVE_STRATEGY:-no}"
export TRAIN_LOGGING_STEPS="${TRAIN_LOGGING_STEPS:-1}"
export EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:---max_steps 20 --no_cuda True}"

OUTPUT_DIR="${UCIT_OUTPUT_ROOT}/flickr_tiny_local"

echo "Running tiny local Flickr30k training"
launch_ucit_task "$FLICKR_TINY_TRAIN_JSON" "$OUTPUT_DIR" 0

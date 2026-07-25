#!/bin/bash
set -euo pipefail

# Conservative UCIT full-schedule launcher for Kaggle 2x T4 16GB.
# Task order: ImageNet-R -> ArxivQA -> VizWiz -> IconQA -> CLEVR -> Flickr30k.

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export DEEPSPEED_INCLUDE="${DEEPSPEED_INCLUDE:-localhost:0,1}"
export TRAIN_LAUNCHER="${TRAIN_LAUNCHER:-deepspeed}"
export TRAIN_PER_DEVICE_BATCH="${TRAIN_PER_DEVICE_BATCH:-1}"
export TRAIN_PER_DEVICE_EVAL_BATCH="${TRAIN_PER_DEVICE_EVAL_BATCH:-1}"
export TRAIN_GRAD_ACCUM_STEPS="${TRAIN_GRAD_ACCUM_STEPS:-16}"
export TRAIN_MODEL_MAX_LENGTH="${TRAIN_MODEL_MAX_LENGTH:-768}"
export TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-2}"
export TRAIN_BF16="${TRAIN_BF16:-False}"
export TRAIN_TF32="${TRAIN_TF32:-False}"
export TRAIN_SAVE_STRATEGY="${TRAIN_SAVE_STRATEGY:-epoch}"
export TRAIN_LOGGING_STEPS="${TRAIN_LOGGING_STEPS:-10}"

# QLoRA-style loading is the safest default for 16GB T4 GPUs.
export EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:---bits 4 --fp16 True --gradient_checkpointing True}"

for task in 1 2 3 4 5 6; do
  echo "========== Starting UCIT Task ${task} =========="
  bash "$SCRIPT_DIR/Task${task}.sh"
  echo "========== Finished UCIT Task ${task} =========="
done

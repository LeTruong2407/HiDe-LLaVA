#!/bin/bash

# Edit this file once, then all UCIT training scripts will use these paths.

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

# Base directory where you store downloaded assets.
# Default to a repo-local folder so helper download scripts create assets
# inside this project unless you explicitly override the path.
export HIDE_ASSETS_ROOT="${HIDE_ASSETS_ROOT:-$REPO_ROOT/hide-llava-assets}"

# Models
export LLAVA_BASE_MODEL="${LLAVA_BASE_MODEL:-$HIDE_ASSETS_ROOT/models/llava-v1.5-7b}"
export CLIP_MODEL="${CLIP_MODEL:-$HIDE_ASSETS_ROOT/models/clip-vit-large-patch14-336}"

# Data roots
export DATA_ROOT="${DATA_ROOT:-$HIDE_ASSETS_ROOT/datasets}"
export INSTRUCTION_ROOT="${INSTRUCTION_ROOT:-$HIDE_ASSETS_ROOT/instructions}"

# Outputs
export UCIT_OUTPUT_ROOT="${UCIT_OUTPUT_ROOT:-$REPO_ROOT/outputs/ucit}"

# Training launch settings
export DEEPSPEED_INCLUDE="${DEEPSPEED_INCLUDE:-localhost:0,1,2,3}"
export MASTER_PORT="${MASTER_PORT:-29601}"
export TRAIN_LAUNCHER="${TRAIN_LAUNCHER:-auto}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-./scripts/zero2.json}"

# Training hyperparameter defaults
export TRAIN_NUM_EPOCHS="${TRAIN_NUM_EPOCHS:-1}"
export TRAIN_PER_DEVICE_BATCH="${TRAIN_PER_DEVICE_BATCH:-24}"
export TRAIN_PER_DEVICE_EVAL_BATCH="${TRAIN_PER_DEVICE_EVAL_BATCH:-16}"
export TRAIN_GRAD_ACCUM_STEPS="${TRAIN_GRAD_ACCUM_STEPS:-1}"
export TRAIN_BF16="${TRAIN_BF16:-True}"
export TRAIN_TF32="${TRAIN_TF32:-True}"
export TRAIN_MODEL_MAX_LENGTH="${TRAIN_MODEL_MAX_LENGTH:-2048}"
export TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-4}"
export TRAIN_SAVE_STRATEGY="${TRAIN_SAVE_STRATEGY:-epoch}"
export TRAIN_LOGGING_STEPS="${TRAIN_LOGGING_STEPS:-1}"
export EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

# Task-specific instruction files
export UCIT_TASK1_TRAIN_JSON="${UCIT_TASK1_TRAIN_JSON:-$INSTRUCTION_ROOT/ImageNet-R/train.json}"
export UCIT_TASK2_TRAIN_JSON="${UCIT_TASK2_TRAIN_JSON:-$INSTRUCTION_ROOT/ArxivQA/train_4w.json}"
export UCIT_TASK3_TRAIN_JSON="${UCIT_TASK3_TRAIN_JSON:-$INSTRUCTION_ROOT/VizWiz/train.json}"
export UCIT_TASK4_TRAIN_JSON="${UCIT_TASK4_TRAIN_JSON:-$INSTRUCTION_ROOT/IconQA/train.json}"
export UCIT_TASK5_TRAIN_JSON="${UCIT_TASK5_TRAIN_JSON:-$INSTRUCTION_ROOT/CLEVR/train_4w.json}"
export UCIT_TASK6_TRAIN_JSON="${UCIT_TASK6_TRAIN_JSON:-$INSTRUCTION_ROOT/Flickr30k/train_brief_4w.json}"
export FLICKR_TINY_TRAIN_JSON="${FLICKR_TINY_TRAIN_JSON:-$INSTRUCTION_ROOT/Flickr30k/train_tiny_32.json}"
export FLICKR_TINY_TEST_JSON="${FLICKR_TINY_TEST_JSON:-$INSTRUCTION_ROOT/Flickr30k/test_tiny_32.json}"
export FLICKR_TINY_VAL_JSON="${FLICKR_TINY_VAL_JSON:-$INSTRUCTION_ROOT/Flickr30k/val_tiny_32.json}"

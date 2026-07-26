#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ] || ! [[ "$1" =~ ^[1-6]$ ]]; then
  echo "Usage: $0 TASK_NUMBER (1-6)" >&2
  exit 2
fi

TASK_NUMBER="$1"
UCIT_TRAIN_SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$UCIT_TRAIN_SCRIPT_DIR/modal_a100_consensus_env.sh"
source "$UCIT_TRAIN_SCRIPT_DIR/../paths.sh"

TASK_IMAGE_PATHS=(
  ""
  "ImageNet-R/train"
  "ArxivQA/images"
  "VizWiz/train"
  "IconQA/iconqa_data/iconqa"
  "CLEVR/images/train"
  "Flickr30k/train"
)
TASK_NAMES=(
  ""
  "ImageNet-R"
  "ArxivQA"
  "VizWiz"
  "IconQA"
  "CLEVR"
  "Flickr30k"
)

TASK_JSON_VARIABLE="UCIT_TASK${TASK_NUMBER}_TRAIN_JSON"
TASK_JSON="${!TASK_JSON_VARIABLE}"
TASK_IMAGE_PATH="${TASK_IMAGE_PATHS[$TASK_NUMBER]}"
TASK_NAME="${TASK_NAMES[$TASK_NUMBER]}"

if [ ! -f "$TASK_JSON" ]; then
  echo "Missing Task $TASK_NUMBER instruction file: $TASK_JSON" >&2
  exit 1
fi

if [ ! -d "$DATA_ROOT/$TASK_IMAGE_PATH" ]; then
  echo "Missing Task $TASK_NUMBER image directory: $DATA_ROOT/$TASK_IMAGE_PATH" >&2
  exit 1
fi

if [ "$TASK_NUMBER" -gt 1 ]; then
  PREVIOUS_TASK=$((TASK_NUMBER - 1))
  PREVIOUS_CHECKPOINT="$UCIT_OUTPUT_ROOT/Task${PREVIOUS_TASK}_llava_lora_ours"
  REQUIRED_PREVIOUS_FILES=(
    "adapter_config.json"
    "adapter_model.bin"
    "non_lora_trainables.bin"
    "consensus_subspaces.pt"
    "consensus_summary.json"
  )

  for filename in "${REQUIRED_PREVIOUS_FILES[@]}"; do
    if [ ! -f "$PREVIOUS_CHECKPOINT/$filename" ]; then
      echo "Missing previous-task checkpoint file: $PREVIOUS_CHECKPOINT/$filename" >&2
      echo "Finish consensus Task $PREVIOUS_TASK before starting Task $TASK_NUMBER." >&2
      exit 1
    fi
  done
fi

echo "========== Starting Modal consensus UCIT Task $TASK_NUMBER: $TASK_NAME =========="
bash "$UCIT_TRAIN_SCRIPT_DIR/Task${TASK_NUMBER}.sh"
echo "========== Finished Modal consensus UCIT Task $TASK_NUMBER: $TASK_NAME =========="

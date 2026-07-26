#!/bin/bash
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "Usage: $0 MODAL_PROFILE TASK_NUMBER [LOCAL_OUTPUT_ROOT]" >&2
  exit 2
fi

PROFILE="$1"
TASK_NUMBER="$2"
LOCAL_OUTPUT_ROOT="${3:-outputs/ucit_consensus_modal_a100}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! [[ "$TASK_NUMBER" =~ ^[1-6]$ ]]; then
  echo "TASK_NUMBER must be between 1 and 6." >&2
  exit 2
fi

REMOTE_PATH="ucit_consensus_modal_a100/Task${TASK_NUMBER}_llava_lora_ours"
LOCAL_PATH="$LOCAL_OUTPUT_ROOT/Task${TASK_NUMBER}_llava_lora_ours"

if [ -e "$LOCAL_PATH" ] && [ ! -d "$LOCAL_PATH" ]; then
  echo "Refusing to download over non-directory path: $LOCAL_PATH" >&2
  echo "Move that file aside first, then rerun this script." >&2
  exit 1
fi

mkdir -p "$LOCAL_PATH"

MODAL_PROFILE="$PROFILE" "$PYTHON_BIN" -m modal volume get \
  --env main \
  hide-llava-outputs \
  "$REMOTE_PATH" \
  "$LOCAL_PATH"

NESTED_PATH="$LOCAL_PATH/Task${TASK_NUMBER}_llava_lora_ours"
if [ -d "$NESTED_PATH" ]; then
  for nested_item in "$NESTED_PATH"/*; do
    [ -e "$nested_item" ] || continue
    mv "$nested_item" "$LOCAL_PATH/"
  done
  rmdir "$NESTED_PATH"
fi

required_files=(
  adapter_config.json
  adapter_model.bin
  non_lora_trainables.bin
  consensus_subspaces.pt
  consensus_summary.json
)

for filename in "${required_files[@]}"; do
  if [ ! -f "$LOCAL_PATH/$filename" ]; then
    echo "Downloaded checkpoint is incomplete. Missing: $LOCAL_PATH/$filename" >&2
    exit 1
  fi
done

echo "Downloaded Task $TASK_NUMBER checkpoint to $LOCAL_PATH"

#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 NEW_MODAL_PROFILE" >&2
  exit 2
fi

PROFILE="$1"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
ASSETS_ROOT="$REPO_ROOT/hide-llava-assets"
OUTPUT_ROOT="$REPO_ROOT/outputs/ucit_consensus_modal_a100"
TASK1_CHECKPOINT="$OUTPUT_ROOT/Task1_llava_lora_ours"

if [ ! -d "$ASSETS_ROOT" ]; then
  echo "Missing local assets directory: $ASSETS_ROOT" >&2
  exit 1
fi

required_task1_files=(
  adapter_config.json
  adapter_model.bin
  non_lora_trainables.bin
  consensus_subspaces.pt
  consensus_summary.json
)

if [ ! -d "$TASK1_CHECKPOINT" ]; then
  echo "Missing local Task 1 checkpoint directory: $TASK1_CHECKPOINT" >&2
  echo "Download Task 1 from the old Modal profile before bootstrapping the new account." >&2
  exit 1
fi

for filename in "${required_task1_files[@]}"; do
  if [ ! -f "$TASK1_CHECKPOINT/$filename" ]; then
    echo "Missing local Task 1 checkpoint file: $TASK1_CHECKPOINT/$filename" >&2
    exit 1
  fi
done

MODAL_PROFILE="$PROFILE" "$PYTHON_BIN" -m modal volume create hide-llava-assets --env main || true
MODAL_PROFILE="$PROFILE" "$PYTHON_BIN" -m modal volume create hide-llava-outputs --env main || true

for asset_path in "$ASSETS_ROOT"/*; do
  [ -e "$asset_path" ] || continue
  asset_name="$(basename "$asset_path")"
  echo "Uploading asset path: $asset_name"
  MODAL_PROFILE="$PROFILE" "$PYTHON_BIN" -m modal volume put \
    --force \
    --env main \
    hide-llava-assets \
    "$asset_path" \
    "/$asset_name"
done

MODAL_PROFILE="$PROFILE" "$PYTHON_BIN" -m modal volume put \
  --force \
  --env main \
  hide-llava-outputs \
  "$TASK1_CHECKPOINT" \
  "ucit_consensus_modal_a100/Task1_llava_lora_ours"

echo "New Modal profile is ready for Task 2: $PROFILE"

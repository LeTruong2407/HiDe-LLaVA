#!/bin/bash
set -euo pipefail

if ! command -v dvc >/dev/null 2>&1; then
  echo "dvc is not installed."
  echo "Install it with:"
  echo "  python -m pip install -r requirements.dvc.txt"
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "git is not installed."
  exit 1
fi

if [ ! -d .git ]; then
  echo "Run this script from the repository root."
  exit 1
fi

if [ ! -d .dvc ]; then
  echo "DVC is not initialized in this repo."
  echo "Initialize and configure a GCS remote first, for example:"
  echo "  bash scripts/dvc/setup_gcs_remote.sh gs://YOUR_BUCKET/hide-llava"
  exit 1
fi

if [ $# -lt 1 ]; then
  cat <<EOF
Usage:
  $0 CHECKPOINT_DIR [REMOTE_NAME]

Examples:
  $0 outputs/ucit_consensus/Task1_llava_lora_ours
  $0 outputs/ucit_consensus/Task6_llava_lora_ours gcs

The checkpoint directory should contain files such as:
  adapter_config.json
  adapter_model.bin
  non_lora_trainables.bin
  consensus_subspaces.pt
  consensus_summary.json
EOF
  exit 1
fi

CHECKPOINT_DIR="${1%/}"
REMOTE_NAME="${2:-}"

if [ ! -d "$CHECKPOINT_DIR" ]; then
  echo "Checkpoint directory not found: $CHECKPOINT_DIR"
  exit 1
fi

if [ ! -f "$CHECKPOINT_DIR/adapter_config.json" ]; then
  echo "Warning: adapter_config.json was not found in $CHECKPOINT_DIR"
fi

if [ ! -f "$CHECKPOINT_DIR/adapter_model.bin" ] && [ ! -f "$CHECKPOINT_DIR/adapter_model.safetensors" ]; then
  echo "Warning: no adapter_model.bin or adapter_model.safetensors found in $CHECKPOINT_DIR"
fi

if [ ! -f "$CHECKPOINT_DIR/consensus_subspaces.pt" ]; then
  echo "Warning: consensus_subspaces.pt was not found in $CHECKPOINT_DIR"
  echo "This may be expected for baseline HiDe, but not for the consensus method."
fi

DVC_FILE="${CHECKPOINT_DIR}.dvc"

echo "Tracking checkpoint with DVC:"
echo "  $CHECKPOINT_DIR"
dvc add "$CHECKPOINT_DIR"

git add "$DVC_FILE" .gitignore

echo
echo "Pushing checkpoint data to DVC remote..."
if [ -n "$REMOTE_NAME" ]; then
  dvc push -r "$REMOTE_NAME" "$DVC_FILE"
else
  dvc push "$DVC_FILE"
fi

echo
echo "DVC push complete for:"
echo "  $CHECKPOINT_DIR"
echo
echo "Commit these metadata files so the checkpoint can be pulled later:"
echo "  git add $DVC_FILE .gitignore"
echo "  git commit -m \"track $(basename "$CHECKPOINT_DIR") checkpoint with DVC\""

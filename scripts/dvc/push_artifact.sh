#!/bin/bash
set -euo pipefail

if ! command -v dvc >/dev/null 2>&1; then
  echo "dvc is not installed."
  echo "Install it with:"
  echo "  python3 -m pip install -r requirements.dvc.txt"
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
  echo "  bash scripts/dvc/setup_gcs_remote.sh gs://YOUR_BUCKET/hide-llava-artifacts"
  exit 1
fi

if [ $# -lt 1 ]; then
  cat <<EOF
Usage:
  $0 ARTIFACT_PATH [REMOTE_NAME]

Examples:
  $0 outputs/ucit_consensus/Task1_llava_lora_ours
  $0 results/UCIT/each_dataset/ImageNet-R/consensus-task1
  $0 outputs/ucit_consensus/Task6_llava_lora_ours gcs
EOF
  exit 1
fi

ARTIFACT_PATH="${1%/}"
REMOTE_NAME="${2:-}"

if [ ! -e "$ARTIFACT_PATH" ]; then
  echo "Artifact path not found: $ARTIFACT_PATH"
  exit 1
fi

DVC_FILE="${ARTIFACT_PATH}.dvc"

echo "Tracking artifact with DVC:"
echo "  $ARTIFACT_PATH"
dvc add --force "$ARTIFACT_PATH"

git add -f "$DVC_FILE" .gitignore

echo
echo "Pushing artifact data to DVC remote..."
if [ -n "$REMOTE_NAME" ]; then
  dvc push -r "$REMOTE_NAME" "$DVC_FILE"
else
  dvc push "$DVC_FILE"
fi

echo
echo "DVC push complete for:"
echo "  $ARTIFACT_PATH"
echo
echo "Commit these metadata files:"
echo "  git add $DVC_FILE .gitignore"
echo "  git commit -m \"track $(basename "$ARTIFACT_PATH") artifact with DVC\""

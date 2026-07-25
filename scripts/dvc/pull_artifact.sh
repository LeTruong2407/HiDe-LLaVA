#!/bin/bash
set -euo pipefail

if ! command -v dvc >/dev/null 2>&1; then
  echo "dvc is not installed."
  echo "Install it with:"
  echo "  python3 -m pip install -r requirements.dvc.txt"
  exit 1
fi

if [ ! -d .git ]; then
  echo "Run this script from the repository root."
  exit 1
fi

if [ $# -lt 1 ]; then
  cat <<EOF
Usage:
  $0 ARTIFACT_PATH_OR_DVC_FILE

Examples:
  $0 outputs/ucit_consensus/Task1_llava_lora_ours
  $0 outputs/ucit_consensus/Task1_llava_lora_ours.dvc
  $0 results/UCIT/each_dataset/ImageNet-R/consensus-task1
EOF
  exit 1
fi

TARGET="${1%/}"
if [[ "$TARGET" != *.dvc ]]; then
  TARGET="${TARGET}.dvc"
fi

if [ ! -f "$TARGET" ]; then
  echo "DVC metadata file not found: $TARGET"
  echo "Pull or checkout the corresponding .dvc file from Git first."
  exit 1
fi

dvc pull "$TARGET"

echo
echo "DVC pull complete for:"
echo "  $TARGET"

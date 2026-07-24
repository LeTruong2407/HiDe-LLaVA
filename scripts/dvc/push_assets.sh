#!/bin/bash
set -euo pipefail

ASSET_PATH="${1:-hide-llava-assets}"

if ! command -v dvc >/dev/null 2>&1; then
  echo "dvc is not installed."
  echo "Install it with:"
  echo "  python -m pip install -r requirements.dvc.txt"
  exit 1
fi

if [ ! -e "$ASSET_PATH" ]; then
  echo "Asset path not found: $ASSET_PATH"
  exit 1
fi

dvc add "$ASSET_PATH"
git add "${ASSET_PATH}.dvc" .gitignore
dvc push

echo
echo "DVC push complete for: $ASSET_PATH"
echo "Recommended next step:"
echo "  git add .dvc/config ${ASSET_PATH}.dvc .gitignore"

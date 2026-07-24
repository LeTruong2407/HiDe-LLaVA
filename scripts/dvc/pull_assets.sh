#!/bin/bash
set -euo pipefail

ASSET_PATH="${1:-hide-llava-assets}"

if ! command -v dvc >/dev/null 2>&1; then
  echo "dvc is not installed."
  echo "Install it with:"
  echo "  python -m pip install -r requirements.dvc.txt"
  exit 1
fi

if [ -f "${ASSET_PATH}.dvc" ]; then
  dvc pull "${ASSET_PATH}.dvc"
else
  dvc pull
fi

echo
echo "DVC pull complete for: $ASSET_PATH"

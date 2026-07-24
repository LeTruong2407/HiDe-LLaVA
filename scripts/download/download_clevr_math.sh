#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

need_cmd huggingface-cli

DEST="${DATA_ROOT}/CLEVR"
RAW_DEST="${HIDE_ASSETS_ROOT}/external_raw/CLEVR"
mkdir -p "$RAW_DEST" "$DEST"

echo "Downloading CLEVR-Math dataset files from Hugging Face into $RAW_DEST"
huggingface-cli download --repo-type dataset dali-does/clevr-math \
  --local-dir "$RAW_DEST" \
  --local-dir-use-symlinks False

echo "Copying CLEVR-Math files into $DEST"
cp -R "$RAW_DEST"/. "$DEST"/

echo "Done: $DEST"

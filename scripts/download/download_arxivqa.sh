#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

need_cmd huggingface-cli

DEST="${DATA_ROOT}/ArxivQA"
RAW_DEST="${HIDE_ASSETS_ROOT}/external_raw/ArxivQA"
mkdir -p "$RAW_DEST" "$DEST"

echo "Downloading ArxivQA dataset files from Hugging Face into $RAW_DEST"
huggingface-cli download --repo-type dataset MMInstruction/ArxivQA \
  --local-dir "$RAW_DEST" \
  --local-dir-use-symlinks False

echo "Copying ArxivQA files into $DEST"
cp -R "$RAW_DEST"/. "$DEST"/

echo "Done: $DEST"

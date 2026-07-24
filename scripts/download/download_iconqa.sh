#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

ARCHIVE_URL="${ICONQA_ARCHIVE_URL:-}"
DEST="${DATA_ROOT}/IconQA"
RAW_DEST="${HIDE_ASSETS_ROOT}/external_raw/IconQA"
ARCHIVE_PATH="${RAW_DEST}/iconqa_data.zip"

mkdir -p "$RAW_DEST" "$DEST"

if [ -z "$ARCHIVE_URL" ]; then
  echo "ICONQA_ARCHIVE_URL is not set."
  echo "Open https://iconqa.github.io/ and copy the current dataset archive URL, then run:"
  echo "  ICONQA_ARCHIVE_URL='https://.../iconqa_data.zip' bash scripts/download/download_iconqa.sh"
  exit 1
fi

echo "Downloading IconQA archive from $ARCHIVE_URL"
download_file "$ARCHIVE_URL" "$ARCHIVE_PATH"

echo "Extracting IconQA into $DEST"
extract_archive "$ARCHIVE_PATH" "$DEST"

echo "Done: $DEST"

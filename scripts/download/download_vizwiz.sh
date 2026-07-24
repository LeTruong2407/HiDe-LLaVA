#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

DEST="${DATA_ROOT}/VizWiz"
RAW_DEST="${HIDE_ASSETS_ROOT}/external_raw/VizWiz"
mkdir -p "$RAW_DEST" "$DEST"

TRAIN_URL="${VIZWIZ_TRAIN_URL:-}"
VAL_URL="${VIZWIZ_VAL_URL:-}"
TEST_URL="${VIZWIZ_TEST_URL:-}"

if [ -z "$TRAIN_URL" ] || [ -z "$VAL_URL" ] || [ -z "$TEST_URL" ]; then
  echo "VizWiz direct archive URLs are not fully configured."
  echo "Open https://vizwiz.org/tasks-and-datasets/image-captioning/ and copy the current image archive URLs, then run:"
  echo "  VIZWIZ_TRAIN_URL='https://...train.zip' \\"
  echo "  VIZWIZ_VAL_URL='https://...val.zip' \\"
  echo "  VIZWIZ_TEST_URL='https://...test.zip' \\"
  echo "  bash scripts/download/download_vizwiz.sh"
  exit 1
fi

for split in train val test; do
  var_name="$(echo "VIZWIZ_${split^^}_URL")"
  url="${!var_name}"
  archive="${RAW_DEST}/${split}.zip"
  echo "Downloading VizWiz ${split} from $url"
  download_file "$url" "$archive"
  mkdir -p "$DEST/${split}"
  extract_archive "$archive" "$DEST/${split}"
done

echo "Done: $DEST"

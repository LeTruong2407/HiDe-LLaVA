#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../HiDe/paths.sh"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "Error: huggingface-cli not found."
  echo "Activate your environment first, e.g. 'conda activate hide-llava'."
  exit 1
fi

DOWNLOAD_MODE="${1:-all}"
DATASET_ROOT="${HIDE_ASSETS_ROOT}/hf_ucit_raw"
mkdir -p "$DATASET_ROOT" "$INSTRUCTION_ROOT" "$DATA_ROOT"

echo "Downloading HaiyangGuo/UCIT dataset files into $DATASET_ROOT"
case "$DOWNLOAD_MODE" in
  all)
    huggingface-cli download --repo-type dataset HaiyangGuo/UCIT \
      --local-dir "$DATASET_ROOT" \
      --local-dir-use-symlinks False
    ;;
  instructions)
    huggingface-cli download --repo-type dataset HaiyangGuo/UCIT \
      --include "UCIT/**" \
      --local-dir "$DATASET_ROOT" \
      --local-dir-use-symlinks False
    ;;
  sample)
    huggingface-cli download --repo-type dataset HaiyangGuo/UCIT \
      --include "CoIN_sample/**" \
      --local-dir "$DATASET_ROOT" \
      --local-dir-use-symlinks False
    ;;
  *)
    echo "Usage: $0 [all|instructions|sample]"
    exit 1
    ;;
esac

copy_if_exists() {
  local src="$1"
  local dst="$2"
  if [ -d "$src" ]; then
    mkdir -p "$dst"
    cp -R "$src"/. "$dst"/
  fi
}

echo "Copying UCIT instruction JSONs into $INSTRUCTION_ROOT"
copy_if_exists "$DATASET_ROOT/UCIT/ArxivQA" "$INSTRUCTION_ROOT/ArxivQA"
copy_if_exists "$DATASET_ROOT/UCIT/CLEVR" "$INSTRUCTION_ROOT/CLEVR"
copy_if_exists "$DATASET_ROOT/UCIT/Flickr30k" "$INSTRUCTION_ROOT/Flickr30k"
copy_if_exists "$DATASET_ROOT/UCIT/IconQA" "$INSTRUCTION_ROOT/IconQA"
copy_if_exists "$DATASET_ROOT/UCIT/ImageNet-R" "$INSTRUCTION_ROOT/ImageNet-R"
copy_if_exists "$DATASET_ROOT/UCIT/VizWiz" "$INSTRUCTION_ROOT/VizWiz"

echo "Copying Hugging Face-hosted images if present"
copy_if_exists "$DATASET_ROOT/UCIT/ImageNet-R" "$DATA_ROOT/ImageNet-R"
copy_if_exists "$DATASET_ROOT/UCIT/Flickr30k" "$DATA_ROOT/Flickr30k"

echo "Done."
echo "Instructions copied under: $INSTRUCTION_ROOT"
echo "HF-hosted images copied under: $DATA_ROOT"
echo "You still need manual downloads for ArxivQA, CLEVR, IconQA, and VizWiz images."

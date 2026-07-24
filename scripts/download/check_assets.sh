#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../HiDe/paths.sh"

check_path() {
  local path="$1"
  if [ -e "$path" ]; then
    echo "[OK]    $path"
  else
    echo "[MISS]  $path"
  fi
}

echo "== Models =="
check_path "$LLAVA_BASE_MODEL"
check_path "$CLIP_MODEL"
check_path "$LLAVA_BASE_MODEL/mm_projector.bin"

echo
echo "== Training JSONs =="
check_path "$UCIT_TASK1_TRAIN_JSON"
check_path "$UCIT_TASK2_TRAIN_JSON"
check_path "$UCIT_TASK3_TRAIN_JSON"
check_path "$UCIT_TASK4_TRAIN_JSON"
check_path "$UCIT_TASK5_TRAIN_JSON"
check_path "$UCIT_TASK6_TRAIN_JSON"

echo
echo "== Dataset folders =="
check_path "$DATA_ROOT/ArxivQA"
check_path "$DATA_ROOT/CLEVR"
check_path "$DATA_ROOT/Flickr30k"
check_path "$DATA_ROOT/IconQA"
check_path "$DATA_ROOT/ImageNet-R"
check_path "$DATA_ROOT/VizWiz"

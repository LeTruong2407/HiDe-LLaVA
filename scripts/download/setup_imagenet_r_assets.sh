#!/bin/bash
set -euo pipefail

DOWNLOAD_SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DOWNLOAD_SCRIPT_DIR/../HiDe/paths.sh"

need_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Error: required command '$cmd' not found."
    exit 1
  fi
}

exists_nonempty() {
  local path="$1"
  [ -e "$path" ] && [ "$(find "$path" -mindepth 1 -print -quit 2>/dev/null || true)" ]
}

extract_imagenet_r_if_needed() {
  local dest="$DATA_ROOT/ImageNet-R"
  local archive="$dest/imagenetr.tar.gz"
  local expected="$dest/train"

  if exists_nonempty "$expected"; then
    echo "[SKIP] ImageNet-R is already extracted: $expected"
    return
  fi

  if [ -f "$archive" ]; then
    echo "[RUN ] Extracting ImageNet-R archive"
    tar -xzf "$archive" -C "$dest"
  else
    echo "[WARN] ImageNet-R archive not found at $archive"
  fi
}

need_cmd huggingface-cli
mkdir -p "$HIDE_ASSETS_ROOT" "$DATA_ROOT" "$INSTRUCTION_ROOT"

if exists_nonempty "$LLAVA_BASE_MODEL" && exists_nonempty "$CLIP_MODEL"; then
  echo "[SKIP] Models already exist"
else
  echo "[RUN ] Downloading LLaVA and CLIP models"
  bash "$DOWNLOAD_SCRIPT_DIR/download_models.sh"
fi

echo "[RUN ] Downloading/copying UCIT ImageNet-R files"
bash "$DOWNLOAD_SCRIPT_DIR/download_ucit_hf.sh" all
extract_imagenet_r_if_needed

echo
echo "[CHECK] ImageNet-R Task 1 assets"
bash "$DOWNLOAD_SCRIPT_DIR/check_assets.sh"

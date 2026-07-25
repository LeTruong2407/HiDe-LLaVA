#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../HiDe/paths.sh"

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

run_or_skip_dir() {
  local label="$1"
  local expected_dir="$2"
  shift 2

  if exists_nonempty "$expected_dir"; then
    echo "[SKIP] $label already exists: $expected_dir"
  else
    echo "[RUN ] $label"
    "$@"
  fi
}

extract_imagenet_r_if_needed() {
  local dest="$DATA_ROOT/ImageNet-R"
  local archive="$dest/imagenetr.tar.gz"
  local expected="$dest/train"

  if [ -d "$expected" ]; then
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

maybe_download_iconqa() {
  if exists_nonempty "$DATA_ROOT/IconQA"; then
    echo "[SKIP] IconQA already exists: $DATA_ROOT/IconQA"
    return
  fi

  if [ -z "${ICONQA_ARCHIVE_URL:-}" ]; then
    echo "[WARN] Skipping IconQA images: ICONQA_ARCHIVE_URL is not set."
    echo "       Get the current archive URL from https://iconqa.github.io/ and rerun with:"
    echo "       ICONQA_ARCHIVE_URL='https://.../iconqa_data.zip' bash scripts/download/setup_all_ucit_assets.sh"
    return
  fi

  echo "[RUN ] Downloading IconQA"
  bash "$SCRIPT_DIR/download_iconqa.sh"
}

maybe_download_vizwiz() {
  if exists_nonempty "$DATA_ROOT/VizWiz"; then
    echo "[SKIP] VizWiz already exists: $DATA_ROOT/VizWiz"
    return
  fi

  if [ -z "${VIZWIZ_TRAIN_URL:-}" ] || [ -z "${VIZWIZ_VAL_URL:-}" ] || [ -z "${VIZWIZ_TEST_URL:-}" ]; then
    echo "[WARN] Skipping VizWiz images: VIZWIZ_TRAIN_URL, VIZWIZ_VAL_URL, and VIZWIZ_TEST_URL are not all set."
    echo "       Get the current URLs from https://vizwiz.org/tasks-and-datasets/image-captioning/ and rerun with:"
    echo "       VIZWIZ_TRAIN_URL='https://...train.zip' VIZWIZ_VAL_URL='https://...val.zip' VIZWIZ_TEST_URL='https://...test.zip' bash scripts/download/setup_all_ucit_assets.sh"
    return
  fi

  echo "[RUN ] Downloading VizWiz"
  bash "$SCRIPT_DIR/download_vizwiz.sh"
}

need_cmd huggingface-cli
mkdir -p "$HIDE_ASSETS_ROOT" "$DATA_ROOT" "$INSTRUCTION_ROOT"

echo "Assets root: $HIDE_ASSETS_ROOT"
echo

run_or_skip_dir "LLaVA base model" "$LLAVA_BASE_MODEL" bash "$SCRIPT_DIR/download_models.sh"
run_or_skip_dir "CLIP model" "$CLIP_MODEL" bash "$SCRIPT_DIR/download_models.sh"

echo "[RUN ] Downloading/copying UCIT instruction files and HF-hosted image data"
bash "$SCRIPT_DIR/download_ucit_hf.sh" all
extract_imagenet_r_if_needed

run_or_skip_dir "ArxivQA images" "$DATA_ROOT/ArxivQA" bash "$SCRIPT_DIR/download_arxivqa.sh"
run_or_skip_dir "CLEVR-Math images" "$DATA_ROOT/CLEVR" bash "$SCRIPT_DIR/download_clevr_math.sh"
maybe_download_iconqa
maybe_download_vizwiz

echo
echo "[CHECK] Final asset status"
bash "$SCRIPT_DIR/check_assets.sh"

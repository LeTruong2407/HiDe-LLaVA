#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../paths.sh"

MODEL_PATH="${1:-$UCIT_OUTPUT_ROOT/flickr_tiny_local}"
DEVICE="${DEVICE:-cpu}"
RESULT_DIR="${RESULT_DIR:-$REPO_ROOT/results/UCIT/each_dataset/Flickr30k/tiny_local}"

if [ ! -f "$FLICKR_TINY_TEST_JSON" ]; then
  echo "Missing tiny Flickr test JSON: $FLICKR_TINY_TEST_JSON"
  exit 1
fi

if [ ! -f "$FLICKR_TINY_VAL_JSON" ]; then
  echo "Missing tiny Flickr val JSON: $FLICKR_TINY_VAL_JSON"
  exit 1
fi

mkdir -p "$RESULT_DIR"

python -m llava.eval.model_answer \
  --model-path "$MODEL_PATH" \
  --model-base "$LLAVA_BASE_MODEL" \
  --question-file "$FLICKR_TINY_TEST_JSON" \
  --image-folder "$DATA_ROOT" \
  --text-tower "$CLIP_MODEL" \
  --answers-file "$RESULT_DIR/merge.jsonl" \
  --num-chunks 1 \
  --chunk-idx 0 \
  --temperature 0 \
  --conv-mode vicuna_v1 \
  --device "$DEVICE"

python -m llava.eval.eval_caption \
  --annotation-file "$FLICKR_TINY_VAL_JSON" \
  --result-file "$RESULT_DIR/merge.jsonl" \
  --output-dir "$RESULT_DIR"

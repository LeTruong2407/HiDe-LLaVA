#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../HiDe/paths.sh"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "Error: huggingface-cli not found."
  echo "Activate your environment first, e.g. 'conda activate hide-llava'."
  exit 1
fi

MODEL_ROOT="${HIDE_ASSETS_ROOT}/models"
mkdir -p "$MODEL_ROOT"

echo "Downloading LLaVA base model to $MODEL_ROOT/llava-v1.5-7b"
huggingface-cli download liuhaotian/llava-v1.5-7b \
  --local-dir "$MODEL_ROOT/llava-v1.5-7b" \
  --local-dir-use-symlinks False

echo "Replacing config with repo-specific config.json"
cp "$REPO_ROOT/config.json" "$MODEL_ROOT/llava-v1.5-7b/config.json"

echo "Downloading CLIP model to $MODEL_ROOT/clip-vit-large-patch14-336"
huggingface-cli download openai/clip-vit-large-patch14-336 \
  --local-dir "$MODEL_ROOT/clip-vit-large-patch14-336" \
  --local-dir-use-symlinks False

echo "Done."
echo "LLAVA_BASE_MODEL=$MODEL_ROOT/llava-v1.5-7b"
echo "CLIP_MODEL=$MODEL_ROOT/clip-vit-large-patch14-336"

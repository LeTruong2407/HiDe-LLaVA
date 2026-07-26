#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODAL_DETACH="${MODAL_DETACH:-1}"
EVAL_STAGE="${EVAL_STAGE:-consensus-modal-a100-task1}"
EVAL_QUANT_MODE="${EVAL_QUANT_MODE:-fp16}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-}"

cmd=(
  "$PYTHON_BIN" -m modal run
)

if [ "$MODAL_DETACH" = "1" ]; then
  cmd+=(--detach)
fi

cmd+=(
  modal_imagenet_r_consensus.py
  --action eval
  --eval-stage "$EVAL_STAGE"
  --eval-quant-mode "$EVAL_QUANT_MODE"
)

if [ -n "$EVAL_MAX_SAMPLES" ]; then
  cmd+=(--eval-max-samples "$EVAL_MAX_SAMPLES")
fi

echo "Launching eval with: ${cmd[*]}"
exec "${cmd[@]}"

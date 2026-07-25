#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ] || ! [[ "$1" =~ ^[1-6]$ ]]; then
  echo "Usage: $0 TASK_NUMBER (1-6)" >&2
  exit 2
fi

TASK_NUMBER="$1"
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/kaggle_consensus_2x16gb_env.sh"

if [ "$TASK_NUMBER" -gt 1 ]; then
  PREVIOUS_TASK=$((TASK_NUMBER - 1))
  PREVIOUS_CHECKPOINT="$UCIT_OUTPUT_ROOT/Task${PREVIOUS_TASK}_llava_lora_ours"
  if [ ! -f "$PREVIOUS_CHECKPOINT/consensus_subspaces.pt" ]; then
    echo "Missing consensus state: $PREVIOUS_CHECKPOINT/consensus_subspaces.pt" >&2
    echo "Run consensus Task $PREVIOUS_TASK successfully before Task $TASK_NUMBER." >&2
    exit 1
  fi
fi

echo "========== Starting consensus UCIT Task ${TASK_NUMBER} =========="
bash "$SCRIPT_DIR/Task${TASK_NUMBER}.sh"
echo "========== Finished consensus UCIT Task ${TASK_NUMBER} =========="

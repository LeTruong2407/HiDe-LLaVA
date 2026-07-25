#!/bin/bash
set -euo pipefail

# Consensus-aware HiDe UCIT schedule for Kaggle 2x T4 16GB.
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/kaggle_consensus_2x16gb_env.sh"

for task in 1 2 3 4 5 6; do
  echo "========== Starting consensus UCIT Task ${task} =========="
  bash "$SCRIPT_DIR/Task${task}.sh"
  echo "========== Finished consensus UCIT Task ${task} =========="
done

#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for task in 1 2 3 4 5 6; do
  bash "$SCRIPT_DIR/run_consensus_task_kaggle_2x16gb.sh" "$task"
done

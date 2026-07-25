#!/bin/bash
set -euo pipefail

# Run only consensus-aware UCIT Task 1 (ImageNet-R).
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/kaggle_consensus_2x16gb_env.sh"

echo "========== Starting consensus UCIT Task 1 =========="
bash "$SCRIPT_DIR/Task1.sh"
echo "========== Finished consensus UCIT Task 1 =========="

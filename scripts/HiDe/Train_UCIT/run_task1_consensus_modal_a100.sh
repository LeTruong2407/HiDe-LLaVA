#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/modal_a100_consensus_env.sh"

echo "========== Starting Modal A100 consensus UCIT Task 1: ImageNet-R =========="
bash "$SCRIPT_DIR/Task1.sh"
echo "========== Finished Modal A100 consensus UCIT Task 1: ImageNet-R =========="

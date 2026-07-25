#!/bin/bash
set -euo pipefail

cat <<EOF
This project now uses DVC only for generated artifacts, not datasets/base models.
Use scripts/download/*.sh for datasets and base models.

Use:
  bash scripts/dvc/push_artifact.sh ARTIFACT_PATH

Examples:
  bash scripts/dvc/push_artifact.sh outputs/ucit_consensus/Task1_llava_lora_ours
  bash scripts/dvc/push_artifact.sh results/UCIT/each_dataset/ImageNet-R/consensus-task1
EOF
exit 1

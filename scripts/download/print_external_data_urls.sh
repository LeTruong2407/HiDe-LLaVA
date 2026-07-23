#!/bin/bash
set -euo pipefail

cat <<'EOF'
External dataset image sources for UCIT:

ArxivQA
  https://huggingface.co/datasets/MMInstruction/ArxivQA/tree/main

CLEVR-Math
  https://huggingface.co/datasets/dali-does/clevr-math/tree/main

IconQA
  https://iconqa.github.io/

VizWiz captioning
  https://vizwiz.org/tasks-and-datasets/image-captioning/

Notes:
- As of July 23, 2026, the HaiyangGuo/UCIT Hugging Face dataset hosts the instruction JSONs and some images
  (notably ImageNet-R and Flickr30k), but the README still points to external sources for several image sets.
- After downloading, place the files under the directories configured in scripts/HiDe/paths.sh.
EOF

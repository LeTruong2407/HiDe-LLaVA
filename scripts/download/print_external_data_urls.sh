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

Helper scripts in this repo:
  bash scripts/download/download_arxivqa.sh
  bash scripts/download/download_clevr_math.sh
  ICONQA_ARCHIVE_URL='https://.../iconqa_data.zip' bash scripts/download/download_iconqa.sh
  VIZWIZ_TRAIN_URL='https://...train.zip' \
  VIZWIZ_VAL_URL='https://...val.zip' \
  VIZWIZ_TEST_URL='https://...test.zip' \
  bash scripts/download/download_vizwiz.sh
EOF

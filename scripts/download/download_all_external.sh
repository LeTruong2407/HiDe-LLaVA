#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[1/4] Downloading ArxivQA"
bash "$SCRIPT_DIR/download_arxivqa.sh"

echo "[2/4] Downloading CLEVR-Math"
bash "$SCRIPT_DIR/download_clevr_math.sh"

echo "[3/4] Downloading IconQA"
bash "$SCRIPT_DIR/download_iconqa.sh"

echo "[4/4] Downloading VizWiz"
bash "$SCRIPT_DIR/download_vizwiz.sh"

echo "Done."

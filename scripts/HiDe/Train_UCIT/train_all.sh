#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python -m pip install -e .
bash "$SCRIPT_DIR/Task1.sh"
python -m pip install -e .
bash "$SCRIPT_DIR/Task2.sh"
python -m pip install -e .
bash "$SCRIPT_DIR/Task3.sh"
python -m pip install -e .
bash "$SCRIPT_DIR/Task4.sh"
python -m pip install -e .
bash "$SCRIPT_DIR/Task5.sh"
python -m pip install -e .
bash "$SCRIPT_DIR/Task6.sh"

#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../paths.sh"

SMOKE_DIR="$INSTRUCTION_ROOT/_smoke"
SMOKE_JSON="$SMOKE_DIR/train_smoke.json"
SMOKE_IMAGE_DIR="$DATA_ROOT/_smoke"
SMOKE_IMAGE="$SMOKE_IMAGE_DIR/smoke.png"

mkdir -p "$SMOKE_DIR"
mkdir -p "$SMOKE_IMAGE_DIR"

if [ -f "$UCIT_TASK1_TRAIN_JSON" ] && [ -f "$DATA_ROOT/ImageNet-R/train/n01843383/sculpture_3.jpg" ]; then
  echo "$UCIT_TASK1_TRAIN_JSON"
  exit 0
fi

FLICKR_IMAGE=""
if [ -d "$DATA_ROOT/Flickr30k/train" ]; then
  FLICKR_IMAGE="$(find "$DATA_ROOT/Flickr30k/train" -type f \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' \) | head -n 1)"
fi

if [ -n "${FLICKR_IMAGE:-}" ]; then
  REL_IMAGE="${FLICKR_IMAGE#$DATA_ROOT/}"
else
  python - <<PY
from PIL import Image
img = Image.new("RGB", (32, 32), color=(120, 180, 240))
img.save("$SMOKE_IMAGE")
PY
  REL_IMAGE="${SMOKE_IMAGE#$DATA_ROOT/}"
fi

cat > "$SMOKE_JSON" <<EOF
[
  {
    "id": "smoke-flickr-1",
    "image": "$REL_IMAGE",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWhat is the object in the image?\nAnswer the question using a single word or phrase."
      },
      {
        "from": "gpt",
        "value": "object"
      }
    ]
  }
]
EOF

echo "$SMOKE_JSON"

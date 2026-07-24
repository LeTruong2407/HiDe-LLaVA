#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../HiDe/paths.sh"

COUNT="${1:-32}"
OUT_DIR="${INSTRUCTION_ROOT}/ImageNet-R"
TRAIN_JSON="${OUT_DIR}/train.json"
TEST_JSON="${OUT_DIR}/test_3000.json"
TINY_TRAIN_JSON="${OUT_DIR}/train_tiny_${COUNT}.json"
TINY_TEST_JSON="${OUT_DIR}/test_tiny_${COUNT}.json"

if [ ! -f "$TRAIN_JSON" ]; then
  echo "Missing training JSON: $TRAIN_JSON"
  exit 1
fi

python - "$TRAIN_JSON" "$TEST_JSON" "$TINY_TRAIN_JSON" "$TINY_TEST_JSON" "$COUNT" "$DATA_ROOT" <<'PY'
import json
import os
import sys

train_json, test_json, tiny_train_json, tiny_test_json, count_str, data_root = sys.argv[1:]
count = int(count_str)

def filter_existing(path, limit):
    if not os.path.exists(path):
        return []
    data = json.load(open(path))
    out = []
    for item in data:
        image = item.get("image")
        if image and os.path.exists(os.path.join(data_root, image)):
            out.append(item)
        if len(out) >= limit:
            break
    return out

tiny_train = filter_existing(train_json, count)
tiny_test = filter_existing(test_json, count)

json.dump(tiny_train, open(tiny_train_json, "w"), indent=2)
json.dump(tiny_test, open(tiny_test_json, "w"), indent=2)

print(f"Wrote {len(tiny_train)} samples to {tiny_train_json}")
print(f"Wrote {len(tiny_test)} samples to {tiny_test_json}")
PY

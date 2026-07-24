#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../HiDe/paths.sh"

COUNT="${1:-32}"
ROOT="${INSTRUCTION_ROOT}/Flickr30k"
TRAIN_JSON="${ROOT}/train_brief_4w.json"
TEST_JSON="${ROOT}/test_3000.json"
VAL_JSON="${ROOT}/val_coco_type_3000.json"
TINY_TRAIN_JSON="${ROOT}/train_tiny_${COUNT}.json"
TINY_TEST_JSON="${ROOT}/test_tiny_${COUNT}.json"
TINY_VAL_JSON="${ROOT}/val_tiny_${COUNT}.json"

python - "$TRAIN_JSON" "$TEST_JSON" "$VAL_JSON" "$TINY_TRAIN_JSON" "$TINY_TEST_JSON" "$TINY_VAL_JSON" "$COUNT" "$DATA_ROOT" <<'PY'
import json
import os
import sys

train_json, test_json, val_json, out_train, out_test, out_val, count_str, data_root = sys.argv[1:]
count = int(count_str)

def keep_existing_list(path, limit):
    data = json.load(open(path))
    out = []
    for item in data:
        image = item.get("image")
        if image and os.path.exists(os.path.join(data_root, image)):
            out.append(item)
        if len(out) >= limit:
            break
    return out

tiny_train = keep_existing_list(train_json, count)
tiny_test = keep_existing_list(test_json, count)

json.dump(tiny_train, open(out_train, "w"), indent=2)
json.dump(tiny_test, open(out_test, "w"), indent=2)

test_ids = {str(item["question_id"]) for item in tiny_test}

val_data = json.load(open(val_json))
images = [img for img in val_data.get("images", []) if os.path.exists(os.path.join(data_root, f'Flickr30k/val/{img["file_name"]}')) and os.path.splitext(img["file_name"])[0] in test_ids]
image_ids = {img["id"] for img in images}
annotations = [ann for ann in val_data.get("annotations", []) if ann.get("image_id") in image_ids]

tiny_val = {
    "images": images,
    "annotations": annotations,
    "info": val_data.get("info", {}),
    "licenses": val_data.get("licenses", []),
}
json.dump(tiny_val, open(out_val, "w"), indent=2)

print(f"Wrote {len(tiny_train)} samples to {out_train}")
print(f"Wrote {len(tiny_test)} samples to {out_test}")
print(f"Wrote {len(images)} images to {out_val}")
PY

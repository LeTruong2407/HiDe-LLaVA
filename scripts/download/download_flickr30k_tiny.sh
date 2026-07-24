#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

need_cmd python

COUNT="${1:-32}"
DATASET_ROOT="${DATA_ROOT}/Flickr30k"
TRAIN_ARCHIVE="${DATASET_ROOT}/train.tar.gz"
VAL_ARCHIVE="${DATASET_ROOT}/val.tar.gz"
TRAIN_JSON="${INSTRUCTION_ROOT}/Flickr30k/train_brief_4w.json"
TEST_JSON="${INSTRUCTION_ROOT}/Flickr30k/test_3000.json"

if [ ! -f "$TRAIN_ARCHIVE" ]; then
  echo "Missing archive: $TRAIN_ARCHIVE"
  exit 1
fi

if [ ! -f "$VAL_ARCHIVE" ]; then
  echo "Missing archive: $VAL_ARCHIVE"
  exit 1
fi

python - "$TRAIN_ARCHIVE" "$VAL_ARCHIVE" "$TRAIN_JSON" "$TEST_JSON" "$DATASET_ROOT" "$COUNT" <<'PY'
import json
import os
import sys
import tarfile

train_archive, val_archive, train_json, test_json, dataset_root, count_str = sys.argv[1:]
count = int(count_str)

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def wanted_images_train(path, limit):
    data = read_json(path)
    seen, out = set(), []
    for item in data:
        image = item.get("image")
        if image and image not in seen:
            seen.add(image)
            out.append(image)
        if len(out) >= limit:
            break
    return out

def wanted_images_test(path, limit):
    data = read_json(path)
    seen, out = set(), []
    for item in data:
        image = item.get("image")
        if image and image not in seen:
            seen.add(image)
            out.append(image)
        if len(out) >= limit:
            break
    return out

def extract_selected(archive_path, rel_paths, dataset_root):
    if not rel_paths:
        return 0, []
    with tarfile.open(archive_path, "r:gz") as tar:
        members = {m.name: m for m in tar.getmembers() if m.isfile()}
        extracted = 0
        missing = []
        for rel_path in rel_paths:
            tail = rel_path.split("Flickr30k/", 1)[-1].lstrip("/")
            split_tail = tail.split("/", 1)[-1] if "/" in tail else tail
            candidates = [
                rel_path,
                tail,
                split_tail,
                f"mnt/ShareDB_1TB/datasets/flickr30k/{tail}",
                f"mnt/ShareDB_1TB/datasets/flickr30k/{split_tail}",
            ]
            member = next((members[c] for c in dict.fromkeys(candidates) if c in members), None)
            if member is None:
                missing.append(rel_path)
                continue
            dest = os.path.join(dataset_root, tail)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            src = tar.extractfile(member)
            if src is None:
                missing.append(rel_path)
                continue
            with src, open(dest, "wb") as out:
                out.write(src.read())
            extracted += 1
    return extracted, missing

train_imgs = wanted_images_train(train_json, count)
test_imgs = wanted_images_test(test_json, count)

train_done, train_missing = extract_selected(train_archive, train_imgs, dataset_root)
test_done, test_missing = extract_selected(val_archive, test_imgs, dataset_root)

print(f"Extracted train images: {train_done}")
print(f"Extracted val images: {test_done}")
if train_missing or test_missing:
    print("Missing files from archives:")
    for item in (train_missing + test_missing)[:20]:
        print(item)
PY

echo "Done."

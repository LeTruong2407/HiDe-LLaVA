#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

need_cmd python

COUNT="${1:-32}"
ARCHIVE_URL="${IMAGENET_R_URL:-https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar}"
ARCHIVE_PATH="${HIDE_ASSETS_ROOT}/downloads/imagenet-r.tar"
TARGET_ROOT="${DATA_ROOT}/ImageNet-R"
TRAIN_JSON="${INSTRUCTION_ROOT}/ImageNet-R/train.json"
TEST_JSON="${INSTRUCTION_ROOT}/ImageNet-R/test_3000.json"

mkdir -p "$(dirname "$ARCHIVE_PATH")" "$TARGET_ROOT"

if [ ! -f "$TRAIN_JSON" ]; then
  echo "Missing instruction JSON: $TRAIN_JSON"
  exit 1
fi

echo "Downloading ImageNet-R archive to $ARCHIVE_PATH"
download_file "$ARCHIVE_URL" "$ARCHIVE_PATH"

echo "Extracting up to $COUNT train images and $COUNT test images into $TARGET_ROOT"
python - "$ARCHIVE_PATH" "$TARGET_ROOT" "$TRAIN_JSON" "$TEST_JSON" "$COUNT" <<'PY'
import json
import os
import sys
import tarfile

archive_path, target_root, train_json, test_json, count_str = sys.argv[1:]
count = int(count_str)

def collect_images(path, limit):
    if not path or not os.path.exists(path):
        return []
    data = json.load(open(path))
    images = []
    seen = set()
    for item in data:
        image = item.get("image")
        if image and image not in seen:
            seen.add(image)
            images.append(image)
        if len(images) >= limit:
            break
    return images

needed = collect_images(train_json, count) + collect_images(test_json, count)
needed = list(dict.fromkeys(needed))

prefix_candidates = ["", "imagenet-r/", "ImageNet-R/"]

with tarfile.open(archive_path, "r") as tar:
    members = {m.name: m for m in tar.getmembers() if m.isfile()}
    extracted = 0
    missing = []
    for rel_path in needed:
        rel_norm = rel_path.replace("\\", "/")
        tail = rel_norm.split("ImageNet-R/", 1)[-1] if "ImageNet-R/" in rel_norm else rel_norm
        tail = tail.lstrip("/")
        candidates = [rel_norm, tail]
        for prefix in prefix_candidates:
            candidates.append(f"{prefix}{tail}")
        member = next((members[c] for c in dict.fromkeys(candidates) if c in members), None)
        if member is None:
            missing.append(rel_path)
            continue
        dest_path = os.path.join(target_root, tail)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        src = tar.extractfile(member)
        if src is None:
            missing.append(rel_path)
            continue
        with src, open(dest_path, "wb") as out:
            out.write(src.read())
        extracted += 1

print(f"Extracted {extracted} files into {target_root}")
if missing:
    print("Missing files from archive:")
    for path in missing[:20]:
        print(path)
    if len(missing) > 20:
        print(f"... and {len(missing) - 20} more")
PY

echo "Done."

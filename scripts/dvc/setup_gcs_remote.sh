#!/bin/bash
set -euo pipefail

if ! command -v dvc >/dev/null 2>&1; then
  echo "dvc is not installed."
  echo "Install it with:"
  echo "  python -m pip install -r requirements.dvc.txt"
  exit 1
fi

if [ $# -lt 1 ]; then
  echo "Usage: $0 gs://YOUR_BUCKET[/optional/prefix] [remote_name]"
  exit 1
fi

REMOTE_URL="$1"
REMOTE_NAME="${2:-gcs}"

if [[ "$REMOTE_URL" != gs://* ]]; then
  echo "Remote URL must start with gs://"
  exit 1
fi

if [ ! -d .git ]; then
  echo "Run this script from the repository root."
  exit 1
fi

if [ ! -d .dvc ]; then
  dvc init
fi

dvc remote add -d "$REMOTE_NAME" "$REMOTE_URL" --force

cat <<EOF

DVC remote configured.

Remote name: $REMOTE_NAME
Remote URL:  $REMOTE_URL

Next, configure credentials using one of these options:

1) Application Default Credentials:
   gcloud auth application-default login

2) Service account key in local-only DVC config:
   dvc remote modify --local $REMOTE_NAME credentialpath /absolute/path/to/service-account.json

Then track and sync assets:
   dvc add hide-llava-assets
   git add hide-llava-assets.dvc .gitignore .dvc/config
   dvc push
EOF

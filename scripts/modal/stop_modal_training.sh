#!/bin/bash
set -euo pipefail

APP_NAME="${1:-hide-llava-imagenet-r-consensus}"
MODAL_ENVIRONMENT="${MODAL_ENVIRONMENT:-main}"

if ! command -v modal >/dev/null 2>&1; then
  echo "modal CLI is not installed or not on PATH."
  echo "Install and authenticate first:"
  echo "  python -m pip install modal"
  echo "  modal setup"
  exit 1
fi

set +e
STOP_OUTPUT="$(modal app stop "$APP_NAME" 2>&1)"
STOP_STATUS="$?"
set -e

if [ "$STOP_STATUS" -eq 0 ]; then
  echo "$STOP_OUTPUT"
  echo "Stopped Modal app: $APP_NAME"
  exit 0
fi

echo "$STOP_OUTPUT"
echo
echo "Could not stop app '$APP_NAME' in Modal environment '$MODAL_ENVIRONMENT'."
echo "This usually means the run already finished/stopped, or it is running in a different Modal environment."
echo
echo "Check active apps with:"
echo "  modal app list"
echo
echo "If you use another environment, try:"
echo "  modal app list --env ENV_NAME"
echo "  MODAL_ENVIRONMENT=ENV_NAME bash scripts/modal/stop_modal_training.sh"
exit "$STOP_STATUS"

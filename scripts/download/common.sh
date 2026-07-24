#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../HiDe/paths.sh"

need_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Error: required command '$cmd' not found."
    exit 1
  fi
}

download_file() {
  local url="$1"
  local out="$2"
  mkdir -p "$(dirname "$out")"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 "$url" -o "$out"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$out" "$url"
  else
    echo "Error: neither curl nor wget is available."
    exit 1
  fi
}

extract_archive() {
  local archive="$1"
  local outdir="$2"
  mkdir -p "$outdir"
  case "$archive" in
    *.zip) unzip -q "$archive" -d "$outdir" ;;
    *.tar.gz|*.tgz) tar -xzf "$archive" -C "$outdir" ;;
    *.tar) tar -xf "$archive" -C "$outdir" ;;
    *)
      echo "Unsupported archive format: $archive"
      exit 1
      ;;
  esac
}

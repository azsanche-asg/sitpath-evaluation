#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
DATA_DIR="$(cd "$SCRIPT_DIR/../../data" && pwd -P)/sdd_mini"
mkdir -p "$DATA_DIR"

if [[ -z "${SDD_MINI_URL:-}" ]]; then
  cat <<'NOTE'
No SDD-mini mirror URL detected.
Set the environment variable SDD_MINI_URL to a tar.gz or zip archive containing the curated subset with the following layout:
  train/<scene>/trajectories.csv
  val/<scene>/trajectories.csv
  test/<scene>/trajectories.csv
You can request access to the SitPath public mirror or create your own by trimming the official Stanford Drone Dataset.
NOTE
  exit 0
fi

ARCHIVE_PATH="$(mktemp)"
cleanup() { rm -f "$ARCHIVE_PATH"; }
trap cleanup EXIT

echo "Downloading SDD-mini from $SDD_MINI_URL"
if command -v curl >/dev/null 2>&1; then
  curl -L "$SDD_MINI_URL" -o "$ARCHIVE_PATH"
elif command -v wget >/dev/null 2>&1; then
  wget -O "$ARCHIVE_PATH" "$SDD_MINI_URL"
else
  echo "Neither curl nor wget found; aborting." >&2
  exit 1
fi

case "$SDD_MINI_URL" in
  *.zip)
    unzip -o "$ARCHIVE_PATH" -d "$DATA_DIR"
    ;;
  *.tar.gz|*.tgz)
    tar -xzf "$ARCHIVE_PATH" -C "$DATA_DIR"
    ;;
  *)
    echo "Unknown archive format. Please provide a .zip or .tar.gz" >&2
    exit 1
    ;;
esac

echo "SDD-mini extracted to $DATA_DIR"

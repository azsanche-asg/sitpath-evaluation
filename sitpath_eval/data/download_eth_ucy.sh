#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../data/eth_ucy" && pwd -P)"
mkdir -p "$ROOT_DIR"

declare -A SCENE_FILES=(
  [ETH]="biwi_eth"
  [HOTEL]="biwi_hotel"
  [UNIV]="students001 students003"
  [ZARA1]="zara01"
  [ZARA2]="zara02"
)

BASE_URL="https://raw.githubusercontent.com/HarshayuGirase/Social-STGCNN/master/datasets"

download_txt() {
  local url="$1"
  local dest="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -L "$url" -o "$dest"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$dest" "$url"
  else
    echo "Neither curl nor wget is available. Install one to continue." >&2
    exit 1
  fi
}

for scene in "${!SCENE_FILES[@]}"; do
  scene_dir="$ROOT_DIR/$scene"
  mkdir -p "$scene_dir"
  csv_out="$scene_dir/trajectories.csv"
  rm -f "$csv_out"
  for file_stub in ${SCENE_FILES[$scene]}; do
    txt_url="$BASE_URL/${scene,,}/$file_stub.txt"
    tmp_file="$(mktemp)"
    echo "Downloading $txt_url"
    download_txt "$txt_url" "$tmp_file"
    python - <<'PY'
import pandas as pd
import sys
import pathlib
csv_path = pathlib.Path(sys.argv[1])
tmp_txt = pathlib.Path(sys.argv[2])
df = pd.read_csv(tmp_txt, delim_whitespace=True, names=["frame_id", "track_id", "x", "y"])
df = df[["track_id", "frame_id", "x", "y"]]
exists = csv_path.exists()
df.to_csv(csv_path, mode='a', header=not exists, index=False)
PY
 "$csv_out" "$tmp_file"
    rm -f "$tmp_file"
  done
  echo "Wrote $csv_out"
done

echo "ETH/UCY download + conversion complete under $ROOT_DIR"

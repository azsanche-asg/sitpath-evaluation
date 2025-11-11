#!/usr/bin/env bash
# Clean ETH/UCY downloader for SitPath evaluation suite
# Compatible with Colab, Linux, and macOS
set -e

# Determine target folder relative to script location
base_dir="$(dirname "$0")/../../data/eth_ucy"
mkdir -p "$base_dir"
cd "$base_dir"

echo "📥 Downloading ETH/UCY trajectories (5 scenes)..."

# ETH dataset
wget -q --show-progress https://raw.githubusercontent.com/HarshayuGirase/Social-STGCNN/master/datasets/eth/biwi_eth.txt -O biwi_eth.txt
wget -q --show-progress https://raw.githubusercontent.com/HarshayuGirase/Social-STGCNN/master/datasets/eth/biwi_hotel.txt -O biwi_hotel.txt

# UCY dataset
wget -q --show-progress https://raw.githubusercontent.com/HarshayuGirase/Social-STGCNN/master/datasets/ucy/zara1.txt -O zara1.txt
wget -q --show-progress https://raw.githubusercontent.com/HarshayuGirase/Social-STGCNN/master/datasets/ucy/zara2.txt -O zara2.txt
wget -q --show-progress https://raw.githubusercontent.com/HarshayuGirase/Social-STGCNN/master/datasets/ucy/univ.txt -O univ.txt

echo "✅ ETH/UCY dataset successfully downloaded to: $base_dir"

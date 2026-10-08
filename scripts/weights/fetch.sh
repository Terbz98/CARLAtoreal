#!/bin/bash
# Pull the published fp16 weights into pix2pixHD/checkpoints/ so the render scripts find
# them. Takes the release tag as its only argument; defaults to the latest release.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/../.." && pwd)
REPO=${REPO:-Terbz98/CARLAtoreal}
TAG=${1:-latest}
CK=$HERE/pix2pixHD/checkpoints

if [ "$TAG" = latest ]; then
  BASE="https://github.com/$REPO/releases/latest/download"
else
  BASE="https://github.com/$REPO/releases/download/$TAG"
fi

for M in carla2real_semantic_v93n; do   # the only model trained from scratch on licensed data; see weights/README.md
  mkdir -p "$CK/$M"
  DST=$CK/$M/latest_net_G.pth
  if [ -s "$DST" ]; then
    echo "  $M already present, skipping"
    continue
  fi
  echo "  fetching $M"
  # --no-check-certificate: this project is developed behind an SSL-intercepting corporate proxy.
  # Drop the flag if you are not.
  wget --no-check-certificate -q --show-progress -O "$DST" "$BASE/${M}_G_fp16.pth"
done
echo "weights are in $CK"

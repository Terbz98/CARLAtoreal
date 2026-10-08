#!/bin/bash
# Write fp16 copies of the publishable generator(s) into weights/, ready to attach to a release.
#
# fp32 is 700 MB per model, which is seven times GitHub's 100 MB blob limit; fp16 is 350 MB, still
# over it but comfortably inside the 2 GB release-asset cap for both models together. See
# weights/README.md for why a release asset rather than a commit.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/../.." && pwd)
CK=${CHECKPOINTS:-$HERE/pix2pixHD/checkpoints}
OUT=$HERE/weights
mkdir -p "$OUT"

for M in carla2real_semantic_v93n; do   # the only model trained from scratch on licensed data; see weights/README.md
  SRC=$CK/$M/latest_net_G.pth
  if [ ! -s "$SRC" ]; then
    echo "missing $SRC -- set CHECKPOINTS to where the trained models live" >&2
    exit 1
  fi
  python3 "$HERE/scripts/weights/export_weights.py" "$SRC" "$OUT/${M}_G_fp16.pth"
done
echo
echo "ready in $OUT -- attach these to a GitHub release, do not git add them without reading"
echo "weights/README.md on the Git LFS bandwidth limit first."

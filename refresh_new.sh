#!/bin/bash
# Organise delivered clips as  results/mp4/NEW/<model>/town<N>/<sunny|night>/
#
# One folder per MODEL VERSION, each with the full 5-town x 2-weather skeleton. Every future
# model (v50, v51, ...) gets its own sibling folder automatically -- the version is read from
# the delivered filename, so nothing here needs editing when a new one lands. Empty weather
# folders are kept on purpose: they show at a glance what a version has not covered.
#
# HARD links, not copies: same filesystem, so no extra disk, they behave as real files to any
# player, and since the delivery scripts `cp` over an existing path (rewriting the inode rather
# than replacing it) they stay current by themselves.
#
# Naming trap this also fixes: in CARLA/ the file suffixed "_visionpilot" is really the PLAIN
# render; the actual Vision Pilot HUD overlay is the same filename under calibrated/.
. "$(dirname "${BASH_SOURCE[0]}")/config.sh"
set -u
SRC=$CARLA2REAL_OUT
NEW=$CARLA2REAL_ROOT/pix2pixHD/results/mp4/NEW
# the coverage job adds scenarios beyond the original five, so they must be listed here or
# their clips are delivered but never appear in the organised tree
TOWNS="03:3 04:4 05:5 06:6 10hd:10 07:7 05dense:05dense"
mkdir -p "$NEW"

link() { [ -s "$1" ] || return 1; ln -f "$1" "$2" 2>/dev/null || cp -f "$1" "$2"; }

# discover every model version present among the delivered clips
VERS=$(ls "$SRC"/town*_{sunny,night}_vp55_*_FINAL_1920_visionpilot.mp4 2>/dev/null \
       | grep -v 'prev_' | grep -oP 'vp55_\K[^_]+' | sort -u)
[ -n "$VERS" ] || { echo "no delivered clips found"; exit 0; }

for V in $VERS; do
  n=0
  for T in $TOWNS; do
    low=town${T%%:*}; num=${T##*:}
    for W in sunny night; do
      d=$NEW/$V/town$num/$W; mkdir -p "$d"
      f=$SRC/${low}_${W}_vp55_${V}_FINAL_1920_visionpilot.mp4
      [ -e "$f" ] || continue
      link "$f" "$d/${low}_${W}_${V}_render.mp4" && n=$((n+1))
      link "$SRC/calibrated/$(basename "$f")" "$d/${low}_${W}_${V}_visionpilot.mp4"
    done
  done
  printf '  %-6s %d/10 clips\n' "$V" "$n"
done

cat > "$NEW/README.txt" <<TXT
Delivered CARLA sim-to-real clips, one folder per model version.

  <model>/town<N>/<sunny|night>/
      *_render.mp4        the generated clip, 1920x960
      *_visionpilot.mp4   the same clip with the Vision Pilot HUD drawn on it

Each version folder carries the full 5-town x 2-weather skeleton. An empty folder means that
model has not produced a clip for that town and lighting -- v49 is a sunny model, v47 and v51
are night models, so each covers only half the tree by design.

  v44  original sunny set (superseded: built with the broken homography)
  v47  night baseline
  v49  sunny baseline, chosen by eye
  v50  sunny, chroma grafted onto v45 -- the fix for v49's detail loss
  v51  night, retrained on 6,826 images instead of 4,156

These are hard links to the canonical files in $CARLA2REAL_OUT/. They use no
extra disk and update in place; deleting one here does not touch the original.

Refresh (runs automatically after each render batch):
  bash $CARLA2REAL_ROOT/refresh_new.sh

Last refreshed: $(date '+%F %H:%M')
TXT
echo "  -> $NEW"

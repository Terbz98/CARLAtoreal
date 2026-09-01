#!/bin/bash
# v51d: give NIGHT the cleanup that sunny already received, tuned for night rather than inherited.
#
# WHY. v50d (sunny) = v50 + vehicle colour + edge-masked unsharp + a gated 3-frame temporal median.
# Night never got any of it, and night is measurably the WORSE case. Delivered clips, frames
# 200-320:
#     v50  sunny town03   luminance 93.5   alt p99 34.86   specks 1801
#     v51  night town03   luminance 40.7   alt p99 49.97   specks 3008
#     v51  night town10hd luminance 44.5   alt p99 66.47   specks 3671
# Night flickers roughly 1.4-1.9x as much as sunny did before its fix, and carries about twice the
# white specks reported on the delivered clips.
#
# WHAT TRANSFERS, AND WHAT DOES NOT -- decided by measurement, not by copying the sunny recipe:
#   DE-SHIMMER      yes. The concern was that the gate (|2x(t)-x(t-1)-x(t+1)| > 1) would rarely
#                   fire on dark, low-contrast frames and the stage would be a no-op. It fires on
#                   75-84% of night pixels versus 72% on sunny, so it applies normally.
#   DESPECKLE       new, night-specific. This is the "faint shiny white dots" complaint, and night
#                   has twice sunny's count. Only tiny, colourless, locally-bright blobs are
#                   replaced, so lamps, headlights and signals are untouched.
#   SHARPEN         NO. Sunny used 0.55, but sharpening amplifies exactly the high-frequency
#                   speckle being removed here, and night is already noisier.
#   VEHICLE COLOUR  NO. It keys on hue, which is close to meaningless on a dark car at night.
#
# Order matters: de-shimmer first (it is temporal and benefits from the specks still being there
# to average against), then despeckle whatever survives.
set -u
. "$(dirname "${BASH_SOURCE[0]}")/config.sh"
BASE=$CARLA2REAL_ROOT
CE="conda run -n carla_env"
DEST=$CARLA2REAL_OUT
VPB=$PERCEPTION_ROOT/build
OUT=$BASE/pix2pixHD/results/mp4/v51d
LOG=$BASE/pix2pixHD/checkpoints/v51d_log.txt
mkdir -p "$OUT" "$DEST/logs_v51d" "$DEST/vp_input_1024"
echo "=== V51D night cleanup $(date) ===" > "$LOG"

for T in town03 town04 town05 town06 town10hd; do
  SRC=$DEST/${T}_night_vp55_v51_FINAL_1920_visionpilot.mp4
  [ -s "$SRC" ] || { echo "  $T: no v51 clip, skipping" >> "$LOG"; continue; }
  DN=${T}_night_vp55_v51d_FINAL_1920_visionpilot.mp4
  if [ -s "$DEST/$DN" ] && [ -s "$DEST/logs_v51d/${T}_night_score.json" ]; then
    echo "  $T already done and scored" >> "$LOG"; continue
  fi
  echo "--- $T $(date)" >> "$LOG"

  if [ -s "$DEST/$DN" ]; then
    echo "  clip exists, scoring only" >> "$LOG"
  else
  # lossless intermediates: repeated mp4v re-encoding measurably costs object detection
  $CE python3 -u $BASE/temporal_deshimmer.py "$SRC" "$OUT/${T}_ds.avi" --alt 1.0 --flow 6.0 >> "$LOG" 2>&1
  A=$OUT/${T}_ds.avi; [ -s "$A" ] || A=$SRC
  $CE python3 -u $BASE/despeckle_night.py "$A" "$OUT/${T}_dsp.avi" >> "$LOG" 2>&1
  B=$OUT/${T}_dsp.avi; [ -s "$B" ] || B=$A

  $CE python3 -u -c "
import cv2
c=cv2.VideoCapture('$B')
w=cv2.VideoWriter('$DEST/$DN',cv2.VideoWriter_fourcc(*'mp4v'),c.get(5) or 30,(1920,960))
n=0
while True:
    ok,f=c.read()
    if not ok: break
    w.write(cv2.resize(f,(1920,960),interpolation=cv2.INTER_AREA)); n+=1
w.release(); print('  FINAL_1920 %d frames'%n)" >> "$LOG" 2>&1
  [ -s "$DEST/$DN" ] || { echo "  $T FAILED" >> "$LOG"; continue; }
  fi

  # Vision Pilot wants 1024x512 -- feeding 1920x960 costs about 4x in lateral accuracy
  $CE python3 -u -c "
import cv2
c=cv2.VideoCapture('$DEST/$DN')
w=cv2.VideoWriter('$DEST/vp_input_1024/$DN',cv2.VideoWriter_fourcc(*'mp4v'),c.get(5) or 30,(1024,512))
while True:
    ok,f=c.read()
    if not ok: break
    w.write(cv2.resize(f,(1024,512),interpolation=cv2.INTER_AREA))
w.release()" >> "$LOG" 2>&1
  # record_carla.sh takes FOUR arguments: input, speed file, output mp4, log path. Redirecting
  # stdout instead of passing the last two makes it exit with "need output mp4" -- which is what
  # happened on the first run and cost every clip its score.
  SPD=$DEST/${T}_night_frame_speed.txt
  ( cd "$VPB" && ./record_carla.sh "$DEST/vp_input_1024/$DN" "$SPD" \
      "$DEST/calibrated/$DN" "$DEST/logs_v51d/${T}_night.log" ) >> "$LOG" 2>&1 \
    || echo "  $T: VP run failed" >> "$LOG"
  G=$DEST/gt/${T}_night_gt.json
  [ -s "$G" ] && $CE python3 $BASE/score_vp.py "$G" "$DEST/logs_v51d/${T}_night.log" \
      --json "$DEST/logs_v51d/${T}_night_score.json" >> "$LOG" 2>&1
  rm -f "$OUT/${T}_ds.avi" "$OUT/${T}_dsp.avi"
  echo "  $T DONE -> $DN" >> "$LOG"
done

$CE python3 $BASE/flicker_report.py v51 v51d --weather night --frames 300 >> "$LOG" 2>&1
bash $BASE/organise_calibrated.sh >> "$LOG" 2>&1
echo "=== V51D DONE $(date) ===" >> "$LOG"

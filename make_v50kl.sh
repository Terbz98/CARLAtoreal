#!/bin/bash
# v50k / v50l: v63's colour grade transferred onto a stable carrier, for all five sunny towns.
#
# WHY. Measured on Town10HD frames 300-360:
#     clip   flicker   colourfulness   sharpness
#     v50d    160.00           18.8       802.7
#     v50j    120.00           18.2       682.9
#     v63     189.00           23.1       674.1
# v63 is 18% flickerier but 23% more colourful, and its vibrancy is colour SEPARATION rather than
# saturation -- its saturation is actually LOWER. Stability is temporal, colour is per-frame, so the
# two are separable. fuse_colour.py smooths the transfer statistics over a temporal window first,
# because v63's own colour drifts twice as fast frame to frame and a per-frame match would import
# exactly the flicker being avoided.
#
# Result on Town10HD: v50l reached flicker 120 and colourfulness 23.4 -- the smoothest AND the most
# colourful of every clip measured, v63 included.
#
# TWO CARRIERS, because they are not equivalent:
#   v50k = onto v50d. Sharper, but v50d predates the vehicle-colour fix, and the grade AMPLIFIES
#          that bug -- its magenta bus goes further into pink. Built for completeness; v50l is the
#          candidate.
#   v50l = onto v50j, which already has the achromatic guard, so the bus is correctly red.
set -u
. "$(dirname "${BASH_SOURCE[0]}")/gpu_wait.sh" 2>/dev/null || true
. "$(dirname "${BASH_SOURCE[0]}")/config.sh"
BASE=$CARLA2REAL_ROOT
CE="conda run -n $CARLA2REAL_ENV"
DEST=$CARLA2REAL_OUT
VPB=$PERCEPTION_ROOT/VisionPilot/build
OUT=$CARLA2REAL_ROOT/pix2pixHD/results/mp4/fuse
LOG=$BASE/pix2pixHD/checkpoints/v50kl_log.txt
mkdir -p "$OUT" "$DEST/vp_input_1024"
echo "=== V50K/V50L $(date) ===" > "$LOG"

# The colour source is v63, and its 4-town delivery may still be running.
while ps -eo args | grep -qE '^bash (/.*/)?render_model\.sh'; do sleep 120; done
echo "=== v63 delivery finished, starting fusion $(date) ===" >> "$LOG"

encode () {  # <src> <dst> <w> <h>
  $CE python3 -u -c "
import cv2,sys
c=cv2.VideoCapture('$1')
w=cv2.VideoWriter('$2',cv2.VideoWriter_fourcc(*'mp4v'),c.get(5) or 30,($3,$4))
n=0
while True:
    ok,f=c.read()
    if not ok: break
    w.write(cv2.resize(f,($3,$4),interpolation=cv2.INTER_AREA)); n+=1
w.release(); print('    %d frames'%n)" >> "$LOG" 2>&1
}

for T in town03 town04 town05 town06 town10hd; do
  SRC63=$DEST/${T}_sunny_vp55_v63_FINAL_1920_visionpilot.mp4
  [ -s "$SRC63" ] || { echo "  $T: no v63 clip -- skipped" >> "$LOG"; continue; }

  for V in k l; do
    case $V in k) CARRIER=v50d ;; l) CARRIER=v50j ;; esac
    CAR=$DEST/${T}_sunny_vp55_${CARRIER}_FINAL_1920_visionpilot.mp4
    TAG=v50$V
    DN=${T}_sunny_vp55_${TAG}_FINAL_1920_visionpilot.mp4
    [ -s "$CAR" ] || { echo "  $T $TAG: no $CARRIER carrier -- skipped" >> "$LOG"; continue; }

    if [ ! -s "$DEST/$DN" ]; then
      echo "--- $T $TAG (colour from v63 onto $CARRIER) $(date)" >> "$LOG"
      A=$OUT/${T}_${TAG}.avi
      $CE python3 -u $BASE/fuse_colour.py "$CAR" "$SRC63" "$A" 1.0 61 >> "$LOG" 2>&1
      [ -s "$A" ] || { echo "  $T $TAG: fusion failed" >> "$LOG"; continue; }
      encode "$A" "$DEST/$DN" 1920 960
      rm -f "$A"                      # lossless intermediates are ~1.3 GB each
    else
      echo "--- $T $TAG already delivered" >> "$LOG"
    fi
    [ -s "$DEST/$DN" ] || { echo "  $T $TAG: encode failed" >> "$LOG"; continue; }

    # CALIBRATE. Vision Pilot is fed 1024x512, never 1920x960 -- at full width lane MAE goes
    # 0.27 -> 0.87 m. It is run, never modified; this project does not own it.
    if [ ! -s "$DEST/calibrated/$DN" ]; then
      mkdir -p "$DEST/logs_${TAG}"
      encode "$DEST/$DN" "$DEST/vp_input_1024/$DN" 1024 512
      while pgrep -x VisionPilot >/dev/null 2>&1; do sleep 15; done
      SPD=$DEST/${T}_sunny_frame_speed.txt
      [ -s "$SPD" ] || { echo "  $T $TAG: no speed file, VP skipped" >> "$LOG"; continue; }
      ( cd "$VPB" && ./record_carla.sh "$DEST/vp_input_1024/$DN" "$SPD" \
          "$DEST/calibrated/$DN" "$DEST/logs_${TAG}/${T}_sunny.log" ) >> "$LOG" 2>&1 \
        || echo "  $T $TAG: VP run failed" >> "$LOG"
      G=$DEST/gt/${T}_sunny_gt.json
      [ -s "$G" ] && $CE python3 $BASE/score_vp.py "$G" "$DEST/logs_${TAG}/${T}_sunny.log" \
          --json "$DEST/logs_${TAG}/${T}_sunny_score.json" >> "$LOG" 2>&1
    fi
    echo "  $T $TAG DONE" >> "$LOG"
  done
done

bash $BASE/organise_calibrated.sh >> "$LOG" 2>&1
for V in v50k v50l; do
  echo "--- calibrated/$V/sunny:" >> "$LOG"
  ls "$DEST/calibrated/$V/sunny" >> "$LOG" 2>&1 || echo "    (nothing)" >> "$LOG"
done
echo "=== V50K/V50L DONE $(date) ===" >> "$LOG"

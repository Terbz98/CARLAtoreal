#!/bin/bash
# v50i: rebuild the sunny baseline chain with the vehicle-colour bug fixed.
#
# THE BUG (found 2026-08-27 from the review note "sunny looks worse than night"). Tracing one frame
# through the pipeline showed the raw render and the v50 delivery are both fine, and the damage
# appears at v50b -- the stage that runs protect_vehicle_colour.py. A red bus became MAGENTA.
#
# Replicating that script's own decision per vehicle region on Town10HD frame 450:
#     CARLA median saturation of the vehicles is 5-31 out of 255 -- effectively neutral grey
#     yet it reports a confident hue near 125 (blue), because on a near-grey surface the hue is
#     decided by tiny channel differences and the circular mean is happy to average them
#     the render has the bus at hue ~9 (red), so the disagreement passes the HUE_TOL gate
#     the script therefore overwrites red with "blue", blends at STRENGTH, and red-toward-blue
#     through RGB is magenta
# A control confirms the logic is right when the colour is real: one region with CARLA saturation
# 93 was correctly left alone.
#
# THE FIX is an achromatic guard (MIN_SAT, default 40). When CARLA's region is neutral, do not
# adopt a hue -- only pull the render's SATURATION toward neutral. That still removes the invented
# warm cast this stage exists to fix (21.3% -> 4.8% of frames originally), without asserting a
# colour that is not in the reference.
#
# Rebuilds v50b then v50d from the untouched v50 deliveries, as v50i/v50i, alongside the originals.
set -u
. "$(dirname "${BASH_SOURCE[0]}")/config.sh"
BASE=$CARLA2REAL_ROOT
CE="conda run -n carla_env"
DEST=$CARLA2REAL_OUT
D=$BASE/datasets
DRR=$D/training_v12_mapillary
OUT=$BASE/pix2pixHD/results/mp4/v50i
LOG=$BASE/pix2pixHD/checkpoints/v50i_log.txt
AMOUNT=${AMOUNT:-0.55}
mkdir -p "$OUT"
echo "=== V50I sunny colour fix $(date) ===" > "$LOG"

for T in town03 town04 town05 town06 town10hd; do
  SRC=$DEST/${T}_sunny_vp55_v50_FINAL_1920_visionpilot.mp4
  [ -s "$SRC" ] || { echo "  $T: no v50 clip" >> "$LOG"; continue; }
  case $T in town10hd) TT=Town10HD ;; *) TT=$(echo "${T^}") ;; esac
  REC=$D/recorded_${TT}_sunny_inst
  PHS=test_${TT}_sunny_inst_gt
  echo "--- $T $(date)" >> "$LOG"

  # the composite stages paint from $REC by frame index; a wrong recording silently paints
  # another drive's content into the clip and every stage still reports success
  if ! $CE python3 $BASE/check_reference.py "$SRC" "$REC/rgb" >> "$LOG" 2>&1; then
    echo "  $T: REFERENCE MISMATCH -- skipping" >> "$LOG"; continue
  fi

  $CE python3 -u $BASE/protect_vehicle_colour.py "$SRC" "$REC/rgb" \
      "$DRR/${PHS}_label" "$OUT/${T}_col.avi" >> "$LOG" 2>&1
  A=$OUT/${T}_col.avi; [ -s "$A" ] || { echo "  $T colour stage failed" >> "$LOG"; continue; }

  # THE NEW STAGE. Buildings are 25% of the frame and the generator invents their facades from a
  # label that says only "building" -- inventing differently each frame, which IS the shimmer.
  # CARLA has the real window grids, from the same camera, identical every frame, and measurably
  # MORE detail than the render (Laplacian variance 2478 vs 1434). Inject its high frequencies and
  # keep the render's lighting, exposure and colour. Detail goes UP (+21% facade detail) and the
  # structure stops reinventing itself -- neither of which a temporal filter can do, because a
  # filter can only remove.
  $CE python3 -u $BASE/protect_buildings.py "$A" "$REC/rgb" \
      "$DRR/${PHS}_label" "$OUT/${T}_bld.avi" 1.0 1.4 >> "$LOG" 2>&1
  [ -s "$OUT/${T}_bld.avi" ] && A=$OUT/${T}_bld.avi

  # edge-masked unsharp, exactly as v50b did: flat regions excluded so it does not amplify noise
  $CE python3 -u -c "
import cv2, numpy as np, sys
sys.path.insert(0,'$BASE')
from vidcodec import fourcc_for
c=cv2.VideoCapture('$A'); fps=c.get(5) or 30
W=int(c.get(3)); H=int(c.get(4))
w=cv2.VideoWriter('$OUT/${T}_sharp.avi', fourcc_for('$OUT/${T}_sharp.avi'), fps, (W,H))
n=0
while True:
    ok,f=c.read()
    if not ok: break
    blur=cv2.GaussianBlur(f,(0,0),1.4)
    sharp=cv2.addWeighted(f,1+$AMOUNT,blur,-$AMOUNT,0)
    g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
    e=cv2.dilate(cv2.Canny(g,40,110),np.ones((3,3),np.uint8),1).astype(np.float32)/255.0
    e=cv2.GaussianBlur(e,(0,0),1.0)[...,None]
    w.write(np.clip(f*(1-e)+sharp*e,0,255).astype(np.uint8)); n+=1
w.release(); print('  sharpened %d frames'%n)" >> "$LOG" 2>&1
  B=$OUT/${T}_sharp.avi; [ -s "$B" ] || B=$A

  # buildings are stable now, so drop their smoothing hard -- the earlier 1.0 with a wide window
  # is what produced "very very blurry, it lost all the details". Cars and road use the tuned
  # values that fixed the car regression (car flicker had gone 31 -> 40 when they were starved).
  BLDG_STRENGTH=0.35 ROAD_STRENGTH=0.5 CAR_STRENGTH=0.85 CAR_FLOW=6.0 WINDOW=3 \
    $CE python3 -u $BASE/class_deshimmer.py "$B" "$DRR/${PHS}_label" "$OUT/${T}_ds.avi" --alt 1.0 >> "$LOG" 2>&1
  C=$OUT/${T}_ds.avi; [ -s "$C" ] || C=$B

  DN=${T}_sunny_vp55_v50i_FINAL_1920_visionpilot.mp4
  $CE python3 -u -c "
import cv2
c=cv2.VideoCapture('$C')
w=cv2.VideoWriter('$DEST/$DN',cv2.VideoWriter_fourcc(*'mp4v'),c.get(5) or 30,(1920,960))
n=0
while True:
    ok,f=c.read()
    if not ok: break
    w.write(cv2.resize(f,(1920,960),interpolation=cv2.INTER_AREA)); n+=1
w.release(); print('  FINAL_1920 %d frames'%n)" >> "$LOG" 2>&1
  [ -s "$DEST/$DN" ] || { echo "  $T FAILED" >> "$LOG"; continue; }

  mkdir -p "$DEST/vp_input_1024" "$DEST/logs_v50i"
  $CE python3 -u -c "
import cv2
c=cv2.VideoCapture('$DEST/$DN')
w=cv2.VideoWriter('$DEST/vp_input_1024/$DN',cv2.VideoWriter_fourcc(*'mp4v'),c.get(5) or 30,(1024,512))
while True:
    ok,f=c.read()
    if not ok: break
    w.write(cv2.resize(f,(1024,512),interpolation=cv2.INTER_AREA))
w.release()" >> "$LOG" 2>&1
  SPD=$DEST/${T}_sunny_frame_speed.txt
  ( cd $PERCEPTION_ROOT/build && ./record_carla.sh "$DEST/vp_input_1024/$DN" \
      "$SPD" "$DEST/calibrated/$DN" "$DEST/logs_v50i/${T}_sunny.log" ) >> "$LOG" 2>&1 \
    || echo "  $T: VP run failed" >> "$LOG"
  G=$DEST/gt/${T}_sunny_gt.json
  [ -s "$G" ] && $CE python3 $BASE/score_vp.py "$G" "$DEST/logs_v50i/${T}_sunny.log" \
      --json "$DEST/logs_v50i/${T}_sunny_score.json" >> "$LOG" 2>&1
  rm -f "$OUT/${T}_col.avi" "$OUT/${T}_bld.avi" "$OUT/${T}_sharp.avi" "$OUT/${T}_ds.avi"
  echo "  $T DONE -> $DN" >> "$LOG"
done

$CE python3 $BASE/flicker_report.py v50d v50i --frames 300 >> "$LOG" 2>&1
bash $BASE/organise_calibrated.sh >> "$LOG" 2>&1
echo "=== V50I DONE $(date) ===" >> "$LOG"

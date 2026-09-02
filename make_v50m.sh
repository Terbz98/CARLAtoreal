#!/bin/bash
# v50m: v50l with the car trails removed, the grade stabilised, and more real detail.
#
# v50l became the sunny baseline on 2026-09-02. Three defects were reported against it, and each
# one is fixed at the stage that actually causes it rather than by grading over the top.
#
# 1. GHOST TRAILS ON MOVING VEHICLES (reported on Town05).
#    Measured: v50j 13.35, v50l 13.42 -- the fusion adds nothing, so the trails are inherited from
#    the carrier. class_deshimmer.py's own table sets car strength 0.4 at flow tolerance 4.0 and
#    warns "independent motion -- keep tight or it trails". v50j overrode both to 0.85 / 6.0 to fix
#    a car-flicker regression, which is precisely the trailing recipe. v50m takes the middle:
#    0.55 / 4.5. Flicker trades roughly 1:1 against ghosting, so this is expected to give back a
#    little car stability. That is the correct trade -- a trail is a visible artefact, a small
#    flicker rise is not.
#
# 2. COLOUR SWITCHING BETWEEN SHADES.
#    The grade is driven by whole-frame Lab statistics, so one large coloured object crossing the
#    view drags the frame mean and re-grades everything. ROBUST=1 uses median/MAD, which a minority
#    of extreme pixels cannot move. SLEW caps the grade's per-frame travel, and the window widens
#    61 -> 91. The window removes jitter; only the slew bounds an excursion.
#
# 3. NOT SHARP ENOUGH.
#    Unsharp goes 0.55 -> 0.75 and CARLA's facade injection 1.0 -> 1.2. Both are guarded: the
#    unsharp is edge-masked so flat regions are excluded, and protect_buildings.py keeps its result
#    only where measured detail actually rises. The road is checked afterwards against the
#    real-photograph ceiling -- sharpening that invents road grain is the failure mode this project
#    has hit before, and road_sky_ceiling.py is the gate.
#
# Chain: v50 delivery -> vehicle colour -> CARLA facades -> unsharp -> de-shimmer -> v63 colour
#        grade -> FINAL_1920 -> 1024x512 -> perception stack -> score.
set -u
. "$(dirname "${BASH_SOURCE[0]}")/config.sh"
BASE=$CARLA2REAL_ROOT
CE="conda run -n $CARLA2REAL_ENV"
DEST=$CARLA2REAL_OUT
D=$BASE/datasets
DRR=$D/training_v12_mapillary
OUT=$BASE/pix2pixHD/results/mp4/${TAG:-v50m}
LOG=$BASE/pix2pixHD/checkpoints/v50m_log.txt
AMOUNT=${AMOUNT:-0.75}
mkdir -p "$OUT"
TOWNS=${TOWNS:-"town03 town04 town05 town06 town10hd"}
# TAG lets an unsharp/de-shimmer sweep run without overwriting the delivery it is being compared
# against. Everything downstream -- delivery name, log dir, score dir -- follows it.
TAG=${TAG:-v50m}
echo "=== V50M $(date) ===" > "$LOG"

for T in $TOWNS; do
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
      "$DRR/${PHS}_label" "$OUT/${T}_bld.avi" 1.2 1.4 >> "$LOG" 2>&1
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
  # is what produced "very very blurry, it lost all the details".
  #
  # CARS ARE THE CHANGE IN v50m. v50j ran 0.85 / 6.0 to recover a car-flicker regression, and the
  # trails reported on Town05 are the cost of it. class_deshimmer's own table says 0.4 / 4.0 and
  # warns that independent motion trails if the gate is loosened. 0.55 / 4.5 sits between the two.
  BLDG_STRENGTH=0.35 ROAD_STRENGTH=0.5 CAR_STRENGTH=${CAR_S:-0.55} CAR_FLOW=${CAR_F:-4.5} WINDOW=3 \
    $CE python3 -u $BASE/class_deshimmer.py "$B" "$DRR/${PHS}_label" "$OUT/${T}_ds.avi" --alt 1.0 >> "$LOG" 2>&1
  C=$OUT/${T}_ds.avi; [ -s "$C" ] || C=$B

  # THE GRADE, last, on a finished carrier -- v63's colour with the statistics made robust and
  # rate-limited. Skipped rather than faked if the v63 render of this town is missing: a clip
  # silently delivered without the grade would look like a regression with no explanation.
  SRC63=$DEST/${T}_sunny_vp55_v63_FINAL_1920_visionpilot.mp4
  if [ -s "$SRC63" ]; then
    ROBUST=1 SLEW=0.08 $CE python3 -u $BASE/fuse_colour.py "$C" "$SRC63" "$OUT/${T}_fuse.avi" 1.0 91 \
      >> "$LOG" 2>&1
    [ -s "$OUT/${T}_fuse.avi" ] && C=$OUT/${T}_fuse.avi || echo "  $T: fusion failed, ungraded" >> "$LOG"
  else
    echo "  $T: no v63 colour source -- delivering UNGRADED" >> "$LOG"
  fi

  DN=${T}_sunny_vp55_${TAG}_FINAL_1920_visionpilot.mp4
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

  mkdir -p "$DEST/vp_input_1024" "$DEST/logs_${TAG}"
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
  ( cd "$PERCEPTION_ROOT/VisionPilot/build" && ./record_carla.sh "$DEST/vp_input_1024/$DN" \
      "$SPD" "$DEST/calibrated/$DN" "$DEST/logs_${TAG}/${T}_sunny.log" ) >> "$LOG" 2>&1 \
    || echo "  $T: VP run failed" >> "$LOG"
  G=$DEST/gt/${T}_sunny_gt.json
  [ -s "$G" ] && $CE python3 $BASE/score_vp.py "$G" "$DEST/logs_${TAG}/${T}_sunny.log" \
      --json "$DEST/logs_${TAG}/${T}_sunny_score.json" >> "$LOG" 2>&1
  rm -f "$OUT/${T}_col.avi" "$OUT/${T}_bld.avi" "$OUT/${T}_sharp.avi" "$OUT/${T}_ds.avi" "$OUT/${T}_fuse.avi"
  echo "  $T DONE -> $DN" >> "$LOG"
done

$CE python3 $BASE/flicker_report.py v50l v50m --frames 300 >> "$LOG" 2>&1
bash $BASE/organise_calibrated.sh >> "$LOG" 2>&1
echo "=== V50M DONE $(date) ===" >> "$LOG"

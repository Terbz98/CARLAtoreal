#!/bin/bash
# v50r: the v66 render, tone-restored and delivered through the v50q chain.
#
# WHY A NEW BASE RENDER AT ALL. v66 is the first model to move the largest measured gap in the
# project. Vegetation detail against real photographs, Town10HD:
#     v50 parent   near  672   far  561   51% of the photo ceiling
#     v66          near 1346   far  859   90% of the photo ceiling
# and its road is SMOOTHER than the parent (0.76x), where v63 -- the last attempt at this gap --
# sat at 1.8-2.0x and cost 5 points of detection recall. The roughness prior gave the generator the
# one thing the label map never told it: how rough a surface is.
#
# TWO THINGS HAD TO BE ESTABLISHED FIRST, both by measurement rather than assumption.
#
# 1. The inference-side channel had to be CALIBRATED. Raw CARLA roughness sits at roughly twice the
#    training median (p50 0.322 against 0.180), so the model was extrapolating -- the same failure
#    that produced v65's crosshatch. Histogram-matching the CARLA maps to the corpus distribution
#    moved road grain from 1.53x the parent to 0.76x and raised the vegetation ceiling 84% -> 90%.
#    gen_texture_energy.py --match-ref. Relative ordering survives: road 0.117, veg 0.794.
#
# 2. The tone shift is NOT the channel's fault. v66 renders roads +35 luminance and cars -28
#    against the parent, which looked disqualifying. But v63 -- a different fine-tune of the same
#    parent, same corpus, same lr, NO new channel -- shows the same pattern at half the size
#    (road +19, cars -13). So fine-tuning v50 on this corpus drifts the tone regardless, which is
#    what the v63/v64 notes already suspected about its road grain. The channel amplifies a drift
#    it did not cause.
#
# THE FIX FOR THE DRIFT is the mechanism that already works here: per-class colour grading. The
# grade is matched to the CURRENT BASELINE's delivered clip, so v50m's proven tone and colour are
# restored class by class -- cars back to car brightness, road back to road brightness -- while
# every bit of v66's structure is kept. This is the v50l/v50q move applied to a better carrier,
# not a new idea.
#
# Chain: v66 delivery -> vehicle colour -> CARLA facades -> unsharp -> de-shimmer -> per-class
#        grade toward v50m -> FINAL_1920 -> 1024x512 -> perception stack -> score.
set -u
. "$(dirname "${BASH_SOURCE[0]}")/../../configs/config.sh"
BASE=$CARLA2REAL_ROOT
CE="conda run -n $CARLA2REAL_ENV"
DEST=$CARLA2REAL_OUT
D=$BASE/datasets
DRR=$D/training_v12_mapillary
OUT=$BASE/pix2pixHD/results/mp4/${TAG:-v50r}
LOG=$BASE/pix2pixHD/checkpoints/v50r_log.txt
AMOUNT=${AMOUNT:-0.75}
mkdir -p "$OUT"
TOWNS=${TOWNS:-"town03 town04 town05 town06 town10hd"}
# TAG lets an unsharp/de-shimmer sweep run without overwriting the delivery it is being compared
# against. Everything downstream -- delivery name, log dir, score dir -- follows it.
TAG=${TAG:-v50r}
echo "=== V50R $(date) ===" > "$LOG"

for T in $TOWNS; do
  SRC=$DEST/${T}_sunny_vp55_${BASE_TAG:-v66}_FINAL_1920_visionpilot.mp4
  [ -s "$SRC" ] || { echo "  $T: no ${BASE_TAG:-v66} clip -- render it first" >> "$LOG"; continue; }
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
  BLDG_STRENGTH=${BLDG_S:-0.35} ROAD_STRENGTH=${ROAD_S:-0.5} VEG_STRENGTH=${VEG_S:-0.5} CAR_STRENGTH=${CAR_S:-0.55} CAR_FLOW=${CAR_F:-4.5} WINDOW=${WIN:-3} \
    $CE python3 -u $BASE/class_deshimmer.py "$B" "$DRR/${PHS}_label" "$OUT/${T}_ds.avi" --alt 1.0 >> "$LOG" 2>&1
  C=$OUT/${T}_ds.avi; [ -s "$C" ] || C=$B

  # THE GRADE, last, on a finished carrier -- v63's colour with the statistics made robust and
  # rate-limited. Skipped rather than faked if the v63 render of this town is missing: a clip
  # silently delivered without the grade would look like a regression with no explanation.
  # the grade RESTORES tone rather than adding vibrancy, so the source is the current
  # baseline delivery, not v63 -- v50m already carries v63's colour, matched per class
  SRC63=$DEST/${T}_sunny_vp55_${COLOUR_SRC:-v50m}_FINAL_1920_visionpilot.mp4
  if [ -s "$SRC63" ]; then
    LABDIR="$DRR/${PHS}_label" ROBUST=1 SLEW=${SLEW_V:-0.08} \
      $CE python3 -u $BASE/fuse_colour.py "$C" "$SRC63" "$OUT/${T}_fuse.avi" 1.0 ${FUSE_WIN:-91} \
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
  ( cd "$VPB" && ./record_carla.sh "$DEST/vp_input_1024/$DN" \
      "$SPD" "$DEST/calibrated/$DN" "$DEST/logs_${TAG}/${T}_sunny.log" ) >> "$LOG" 2>&1 \
    || echo "  $T: VP run failed" >> "$LOG"
  G=$DEST/gt/${T}_sunny_gt.json
  [ -s "$G" ] && $CE python3 $BASE/score_vp.py "$G" "$DEST/logs_${TAG}/${T}_sunny.log" \
      --json "$DEST/logs_${TAG}/${T}_sunny_score.json" >> "$LOG" 2>&1
  rm -f "$OUT/${T}_col.avi" "$OUT/${T}_bld.avi" "$OUT/${T}_sharp.avi" "$OUT/${T}_ds.avi" "$OUT/${T}_fuse.avi"
  echo "  $T DONE -> $DN" >> "$LOG"
done

$CE python3 $BASE/flicker_report.py v50m v50r --frames 300 >> "$LOG" 2>&1
bash $BASE/organise_calibrated.sh >> "$LOG" 2>&1
echo "=== V50R DONE $(date) ===" >> "$LOG"

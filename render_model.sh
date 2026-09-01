#!/bin/bash
# Render an already-trained model over towns whose inference channels already exist, then
# deliver and score it.  usage: render_model.sh <sunny|night> <model_name> <tag> <town...>
#
# Phase 2 of the 72-hour queue. All the expensive prep (labels, enriched labels, instance-merged
# edges, MoGe depth+normal, chroma, light) was built by the v49 sunny and night chains, so this
# only re-runs what actually depends on the model weights.
#
# New models are delivered ALONGSIDE the current ones under their own tag, never over them.
# The sunny baseline was chosen by eye and that call belongs to the reviewer, so nothing replaces a
# delivered clip automatically -- both sit in CARLA/ and score_vp.py reports the difference.
set -u
W=${1:?sunny|night}; M=${2:?model name}; TAG=${3:?tag e.g. v50}; shift 3
. "$(dirname "${BASH_SOURCE[0]}")/config.sh"
BASE=$CARLA2REAL_ROOT
CE="conda run -n carla_env"
D=$BASE/datasets
DRR=$D/training_v12_mapillary
CK=$BASE/pix2pixHD/checkpoints
TMP=${TMPDIR:-/tmp}/carla2real
R=$BASE/pix2pixHD/results
VPD=$R/mp4/vision_pilot
DEST=$CARLA2REAL_OUT
VPB=$PERCEPTION_ROOT/build
LOG=$CK/render_${TAG}_log.txt
mkdir -p "$TMP"
ARCH_S="--label_nc 65 --no_instance --edge_input --depth_input --normal_input \
--netG local --ngf 32 --n_downsample_global 4 --n_local_enhancers 1 --n_blocks_local 9"
if [ "$W" = night ]; then ARCH="$ARCH_S --light_input"; SUF=night_inst
else ARCH="$ARCH_S --chroma_input"; SUF=sunny_inst; fi
# TEMPORAL=1 for the vid2vid-style models: --temporal widens netG by output_nc (test.py feeds the
# previous GENERATED frame back in, autoregressively). Omitting it here would build a narrower
# netG than the checkpoint, and load_network would silently drop the first conv to random init.
[ "${TEMPORAL:-0}" = 1 ] && ARCH="$ARCH --temporal"

echo "=== RENDER $TAG ($M, $W, epoch ${EPOCH:-latest}) $(date) ===" > "$LOG"
[ -s "$CK/$M/latest_net_G.pth" ] || { echo "  no checkpoint $M -- abort" >> "$LOG"; exit 1; }

for T in "$@"; do
  low=$(echo "$T" | tr 'A-Z' 'a-z')
  NAME=${T}_${SUF}; PHS=test_${NAME}_gt; REC=$D/recorded_$NAME
  OUT=$VPD/$low; mkdir -p "$OUT"; B=$OUT/${low}_${W}_${TAG}
  [ -s "${B}_FINAL_1920.mp4" ] && { echo "=== $T already done ===" >> "$LOG"; continue; }

  # every channel this arch consumes must be present and complete, or the render silently
  # misaligns -- the failure that produced a whole tumbling Town10HD clip from stale channels
  need="label edge depth normal"; [ "$W" = night ] && need="$need light" || need="$need chroma"
  nf=$(ls "$REC/rgb" 2>/dev/null | wc -l); bad=0
  # a missing recording gives nf=0, which would satisfy every "channel >= nf" test at 0>=0 and
  # then satisfy "rendered >= nf-10" at 0>=-10 -- an empty render sailing into post-processing
  [ "$nf" -ge 100 ] || { echo "  $T: recording has $nf frames -- SKIPPING town" >> "$LOG"; continue; }
  for c in $need; do
    k=$(ls "$DRR/${PHS}_$c" 2>/dev/null | wc -l)
    [ "$k" -ge "$nf" ] || { echo "  $T: channel $c has $k of $nf -- SKIPPING town" >> "$LOG"; bad=1; }
  done
  [ "$bad" = 1 ] && continue
  echo "=== $T render ($nf frames) $(date) ===" >> "$LOG"

  # honour EPOCH: test.py writes ${PHS}_<epoch>/images, so a hardcoded _latest here
  # counts 0 frames for any other epoch and fails the completeness gate silently
  DD=$R/$M/${PHS}_${EPOCH:-latest}/images
  rm -rf "$R/$M/${PHS}_${EPOCH:-latest}"
  ( cd $BASE/pix2pixHD && $CE python3 -u test.py --name $M --dataroot "$DRR" $ARCH \
      --loadSize 2048 --resize_or_crop scale_width --phase $PHS --how_many 99999 \
      --which_epoch "${EPOCH:-latest}" --gpu_ids 0 ) >> "$LOG" 2>&1
  n=$(ls $DD/*_synthesized_image.jpg 2>/dev/null | wc -l)
  echo "  rendered $n frames" >> "$LOG"
  # a shape disagreement between arch flags and checkpoint does not raise -- it random-inits and
  # renders confident garbage. Catch it here rather than at visual review.
  grep -q 'not initialized' "$LOG" && { echo "  $T: !! ARCH/CHECKPOINT MISMATCH -- aborting" >> "$LOG"; exit 1; }
  [ "$n" -ge $((nf - 10)) ] || { echo "  RENDER FAILED $T ($n/$nf)" >> "$LOG"; continue; }

  # the post stages composite from $REC/rgb by frame index; a wrong recording paints another
  # drive's content over this clip and every stage still reports success
  if ! $CE python3 $BASE/check_reference.py "$DD" "$REC/rgb" >> "$LOG" 2>&1; then
    echo "  $T: REFERENCE MISMATCH -- SKIPPING town" >> "$LOG"; continue
  fi
  echo "=== $T post $(date) ===" >> "$LOG"
  # NO_TEMPORAL_POST=1 -- for models that are temporally consistent BY TRAINING (v54/v55), skip
  # every stage whose only job is to smooth flicker afterwards. Both stages cost detail, and DVP
  # alone costs 1 h 56 per clip. stabilize_frames_v2 is still run because it also does the detail
  # and saturation grade, but with --alpha 0 its temporal blend term goes to zero
  # (eff = alpha * trust * motion_scale), leaving the per-frame work intact.
  SALPHA=0.6; [ "${NO_TEMPORAL_POST:-0}" = 1 ] && SALPHA=0
  $CE python3 -u $BASE/stabilize_frames_v2.py --frames_dir "$DD" --out "${B}_baseline.avi" \
    --alpha $SALPHA --detail_sigma 3 --saturation 0.9 >> "$LOG" 2>&1
  S=${B}_baseline.avi
  if [ "${NO_TEMPORAL_POST:-0}" = 1 ]; then
    echo "  $T: NO_TEMPORAL_POST -- skipping DVP (model is temporally trained)" >> "$LOG"
  else
  PROC=$TMP/${NAME}_${TAG}_proc; OD=$TMP/${NAME}_${TAG}_dvp; rm -rf "$PROC" "$OD"; mkdir -p "$PROC"
  $CE python3 -c "
import glob,shutil
for i,f in enumerate(sorted(glob.glob('$DD/*_synthesized_image.jpg'))): shutil.copy(f,'$PROC/%05d.jpg'%i)" >> "$LOG" 2>&1
  ( cd $BASE/dvp_pytorch && CUDA_VISIBLE_DEVICES=0 $CE python3 main_IRT.py --input "$REC/rgb" \
      --processed "$PROC" --with_IRT 0 --init_features 64 --max_epoch 20 --save_freq 20 \
      --model ${NAME}_${TAG}_dvp --output "$OD" ) >> "$LOG" 2>&1
  $CE python3 $BASE/make_v33.py "${B}_baseline.avi" "$OD" "$B" "VP" >> "$LOG" 2>&1
  [ -s "${B}_v33_sunny.avi" ] && S=${B}_v33_sunny.avi
  rm -rf "$PROC" "$OD"
  fi

  GRAIN_SCALE=0 BLOOM_SCALE=0 CHROMA_SCALE=0 \
    $CE python3 $BASE/photoreal_post.py "$S" "${B}_photoreal.avi" 0.7 >> "$LOG" 2>&1
  P=${B}_photoreal.avi; [ -s "$P" ] || P=$S
  # night only: composite CARLA's own lamp pools back in, or the model fills unlit areas with
  # black voids. Measured cost is real (-11 pts of lead-vehicle detection on town04) but the
  # voids are worse; revisit once v51's extra night data reduces the model's reliance on it.
  if [ "$W" = night ]; then
    $CE python3 -u $BASE/protect_light_pools.py "$P" "$REC/rgb" "$DRR/${PHS}_label" \
      "${B}_pools.avi" 0.8 18 >> "$LOG" 2>&1
    [ -s "${B}_pools.avi" ] && P=${B}_pools.avi
  fi
  $CE python3 -u $BASE/protect_lane_markings.py "$P" "$REC/rgb" "$DRR/${PHS}_label" "${B}_lane.avi" 45 0.9 >> "$LOG" 2>&1
  C=${B}_lane.avi; [ -s "$C" ] || C=$P
  $CE python3 -u $BASE/protect_billboards.py "$C" "$REC/rgb" "$DRR/${PHS}_label_rich" "${B}_bb.avi" 0.85 3 >> "$LOG" 2>&1
  C2=${B}_bb.avi; [ -s "$C2" ] || C2=$C
  TL_CLEAN=1 $CE python3 -u $BASE/protect_traffic_lights_carla.py "$C2" "$REC/rgb" "$DRR/${PHS}_label" \
    "${B}_FINAL.avi" 2 1.0 >> "$LOG" 2>&1
  F=${B}_FINAL.avi; [ -s "$F" ] || F=$C2
  $CE python3 -u -c "
import cv2
c=cv2.VideoCapture('$F')
w=cv2.VideoWriter('${B}_FINAL_1920.mp4',cv2.VideoWriter_fourcc(*'mp4v'),c.get(5) or 30,(1920,960))
n=0
while True:
    ok,f=c.read()
    if not ok: break
    w.write(cv2.resize(f,(1920,960),interpolation=cv2.INTER_AREA)); n+=1
w.release(); print('  FINAL_1920 %d frames'%n)" >> "$LOG" 2>&1
  [ -s "${B}_FINAL_1920.mp4" ] || { echo "  $T CLIP FAILED" >> "$LOG"; continue; }

  # ---- deliver alongside, never over ----
  DN=${low}_${W}_vp55_${TAG}_FINAL_1920_visionpilot.mp4
  cp "${B}_FINAL_1920.mp4" "$DEST/$DN"
  SPD=$DEST/${low}_${W}_frame_speed.txt
  [ -f "$REC/frame_speed.txt" ] && cp "$REC/frame_speed.txt" "$SPD"
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
  ( cd "$VPB" && ./record_carla.sh "$DEST/vp_input_1024/$DN" "$SPD" \
      "$DEST/calibrated/$DN" "$DEST/logs_${TAG}/${low}_${W}.log" ) >> "$LOG" 2>&1 \
    && echo "  VP ok -> calibrated/$DN" >> "$LOG" || echo "  VP FAILED $T" >> "$LOG"

  G=$DEST/gt/${low}_${W}_gt.json
  [ -s "$G" ] && $CE python3 $BASE/score_vp.py "$G" "$DEST/logs_${TAG}/${low}_${W}.log" \
      --json "$DEST/logs_${TAG}/${low}_${W}_score.json" >> "$LOG" 2>&1

  for st in baseline v33_sunny v33_sharp photoreal pools lane bb FINAL; do rm -f "${B}_${st}.avi" "${B}_${st}.mp4"; done
  echo "  $T DELIVERED as $DN" >> "$LOG"
done
# fold the new clips into results/mp4/NEW/<model>/town<N>/<weather>/ -- the version folder
# is created from the filename, so nothing here needs changing when a new model lands
bash $BASE/refresh_new.sh >> "$LOG" 2>&1
echo "=== RENDER $TAG DONE $(date) ===" >> "$LOG"

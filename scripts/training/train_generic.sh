#!/bin/bash
# One training run, fully specified by environment variables, so the autopilot can launch any
# configuration decide_next.py chooses without a bespoke script per experiment.
#
# Required: NAME PARENT DATA EXTRA NITER DECAY LR PHASE
#
# The guards are the same ones every hand-written script in this project carries, because each
# corresponds to a run that was wasted:
#   - the corpus must be complete AND contain the data it claims to (a self-consistent corpus with
#     none of the new data passed every count check once)
#   - a channel-count change needs a graft, or load_network drops the first conv to random init
#     without raising, and the run silently becomes a from-scratch train (this cost v49 its detail)
#   - 'not initialized' in the log means exactly that happened; the run is void, not merely worse
set -u
. "$(dirname "${BASH_SOURCE[0]}")/../../configs/config.sh"
BASE=$CARLA2REAL_ROOT
PY=python3   # NOT conda run: it buffers stdout to the end
CE="conda run -n $CARLA2REAL_ENV"
CK=$BASE/pix2pixHD/checkpoints
DRR=$CARLA2REAL_DATA/training_v12_mapillary
DST=$CARLA2REAL_DATA/$DATA
LOG=$CK/${NAME}_log.txt
ARCH="--label_nc 65 --no_instance --edge_input --depth_input --normal_input $EXTRA \
--netG local --ngf 32 --n_downsample_global 4 --n_local_enhancers 1 --n_blocks_local 9"

echo "=== $NAME  parent=$PARENT  data=$DATA  extra=$EXTRA  lr=$LR  $(date) ===" > "$LOG"
[ -n "${NOTE:-}" ] && echo "  rationale: $NOTE" >> "$LOG"

N=$(ls "$DST/train_img" 2>/dev/null | wc -l)
[ "$N" -gt 5000 ] || { echo "  ABORT: corpus $DATA has $N images" >> "$LOG"; exit 1; }
CHANNELS="train_label train_edge train_depth train_normal"
case "$EXTRA" in *chroma*) CHANNELS="$CHANNELS train_chroma" ;; esac
case "$EXTRA" in *light*)  CHANNELS="$CHANNELS train_light" ;; esac
case "$EXTRA" in *texture*) CHANNELS="$CHANNELS train_texture" ;; esac
for d in $CHANNELS; do
  M=$(ls "$DST/$d" 2>/dev/null | wc -l)
  [ "$M" -eq "$N" ] || { echo "  ABORT: $d has $M of $N -- pairs would misalign" >> "$LOG"; exit 1; }
done
echo "  corpus $DATA verified: $N pairs, channels: $CHANNELS" >> "$LOG"
echo "  composition: $(ls "$DST/train_img" | sed 's/[0-9].*//' | sort | uniq -c | tr '\n' ' ')" >> "$LOG"

# Does this configuration need more input channels than the parent has? If so, graft rather than
# let load_network silently discard the first conv.
#
# ASK THE PARENT HOW WIDE IT IS -- never guess from its NAME. An earlier version decided whether the
# parent carried the roughness prior by looking for "tex" in the parent's name. A parent that carried
# it under another name was then grafted 74 -> 75 channels, the model built 74, and load_network
# silently random-initialised the input layers of the generator and all three discriminators. That
# run trained for ten hours and was void. The arithmetic below mirrors pix2pixHD_model.py exactly:
# normal and chroma are THREE channels each; edge, depth, light, texture and shadow are one.
WANT_IN=65
case "$ARCH$EXTRA" in *edge_input*)    WANT_IN=$((WANT_IN + 1)) ;; esac
case "$ARCH$EXTRA" in *depth_input*)   WANT_IN=$((WANT_IN + 1)) ;; esac
case "$ARCH$EXTRA" in *light_input*)   WANT_IN=$((WANT_IN + 1)) ;; esac
case "$ARCH$EXTRA" in *chroma_input*)  WANT_IN=$((WANT_IN + 3)) ;; esac
case "$ARCH$EXTRA" in *normal_input*)  WANT_IN=$((WANT_IN + 3)) ;; esac
case "$ARCH$EXTRA" in *texture_input*) WANT_IN=$((WANT_IN + 1)) ;; esac
case "$ARCH$EXTRA" in *shadow_input*)  WANT_IN=$((WANT_IN + 1)) ;; esac
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PARENT_W=$($PY "$HERE/read_conv_in.py" "$CK/$PARENT/latest_net_G.pth" 2>/dev/null)
[ -n "${PARENT_W:-}" ] || { echo "  ABORT: cannot read the first-conv width of $PARENT" >> "$LOG"; exit 1; }
NEW_CH=$((WANT_IN - PARENT_W))
[ "$NEW_CH" -ge 0 ] || { echo "  ABORT: parent is $PARENT_W wide, config builds $WANT_IN; a wider parent cannot be grafted down" >> "$LOG"; exit 1; }
echo "  channels: parent $PARENT_W, config $WANT_IN, graft $NEW_CH" >> "$LOG"
INIT=$CK/${NAME}_init
rm -rf "$INIT"; mkdir -p "$INIT"
if [ "$NEW_CH" -gt 0 ]; then
  $CE python3 -u "$HERE/make_v50_init.py" "$CK/$PARENT" "$INIT" --new-channels "$NEW_CH" >> "$LOG" 2>&1 \
    || { echo "  ABORT: graft failed" >> "$LOG"; exit 1; }
else
  cp "$CK/$PARENT"/latest_net_*.pth "$INIT/" 2>/dev/null \
    || { echo "  ABORT: no parent weights" >> "$LOG"; exit 1; }
fi
# re-read what will ACTUALLY be loaded and require an exact match, before a single epoch
GOT=$($PY "$HERE/read_conv_in.py" "$INIT/latest_net_G.pth" 2>/dev/null)
[ "${GOT:-0}" -eq "$WANT_IN" ] || { echo "  ABORT: init is ${GOT:-?} wide, config builds $WANT_IN" >> "$LOG"; exit 1; }

cd $BASE/pix2pixHD
ok=0
for res in "2048 1024" "1536 768" "1024 512"; do
  set -- $res
  echo "--- attempt at loadSize $1 fineSize $2 $(date)" >> "$LOG"
  MARK=$(wc -l < "$LOG")
  $PY -u train.py --name "$NAME" --dataroot "$DST" $ARCH \
    --num_D 3 --lambda_feat 25 --loadSize $1 --fineSize $2 \
    --resize_or_crop scale_width_and_crop --load_pretrain "$INIT" \
    --niter $NITER --niter_decay $DECAY --lr $LR \
    --save_epoch_freq 1 --batchSize 1 --gpu_ids 0 >> "$LOG" 2>&1 && ok=1
  if tail -n +$MARK "$LOG" | grep -q 'not initialized'; then
    echo "!!! parent weights DISCARDED -- this run is void, not merely worse" >> "$LOG"; ok=0; break
  fi
  [ "$ok" = 1 ] && break
  tail -n +$MARK "$LOG" | grep -qi 'out of memory' || break
  echo "  OOM at $1/$2, stepping down" >> "$LOG"; sleep 20
done
cd $BASE
[ "$ok" = 1 ] || { echo "=== $NAME TRAIN FAILED $(date) ===" >> "$LOG"; exit 1; }
echo "=== $NAME TRAINED $(date) ===" >> "$LOG"

echo "=== render $PHASE and score $(date) ===" >> "$LOG"
rm -rf "$BASE/pix2pixHD/results/$NAME/${PHASE}_latest"
( cd $BASE/pix2pixHD && $PY -u test.py --name $NAME --dataroot "$DRR" $ARCH \
    --loadSize 2048 --resize_or_crop scale_width --phase $PHASE --how_many 400 \
    --which_epoch latest --gpu_ids 0 ) >> "$LOG" 2>&1
D=$BASE/pix2pixHD/results/$NAME/${PHASE}_latest/images
NF=$(ls $D/*_synthesized_image.jpg 2>/dev/null | wc -l)
echo "  rendered $NF frames" >> "$LOG"
[ "$NF" -lt 100 ] && { echo "  RENDER FAILED" >> "$LOG"; exit 1; }
$CE python3 -m carla2real.evaluation.eval_model "$NAME" "$PARENT" "${PHASE}_latest" \
    --out $CK/experiments.json >> "$LOG" 2>&1
echo "=== $NAME DONE $(date) ===" >> "$LOG"

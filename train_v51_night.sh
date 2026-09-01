#!/bin/bash
# v51 -- the night model, retrained on real night data instead of four dashcam drives.
#
# DIAGNOSIS. v47_night trained on 4,156 images: 3,600 are frames from NuRec videos 10-13,
# 534 are v19, and 22 are Mapillary photographs. The sunny lineage trained on 32,475 across
# three datasets. That is an 8x gap in count and far more in scene diversity, and it is the
# best explanation for night looking soft next to sunny.
#
# build_night_corpus_v4.py's own notes anticipated this: "Real night photographs would still be
# welcome for variety ... NuRec 10-13 is all the night data we have." Dark Zurich is that
# variety -- 1920x1080 driving footage, filtered by the same sky-based night test.
#
# SCOPE. Architecture is byte-identical to v47_night. No new channel, no changed loss. The only
# difference is the corpus, so if v51 beats v47 the data is why. Those notes also record what
# happens when the night corpus gets contaminated with daytime images (sky went to the day mode,
# buildings to the night mode), so the same sky test gates every image admitted here.
set -u
. "$(dirname "${BASH_SOURCE[0]}")/config.sh"
BASE=$CARLA2REAL_ROOT
CE="conda run -n $CARLA2REAL_ENV"
D=$CARLA2REAL_DATA
CK=$BASE/pix2pixHD/checkpoints
SRC=$D/training_v47_night
DZ=$D/training_dz_night
DST=$D/training_v51_night
NAME=carla2real_semantic_v51_night
BASEM=carla2real_semantic_v47_night
LOG=$CK/v51_night_log.txt
SUBS="train_img train_label train_edge train_depth train_normal train_light"
ARCH="--label_nc 65 --no_instance --edge_input --depth_input --normal_input --light_input \
--netG local --ngf 32 --n_downsample_global 4 --n_local_enhancers 1 --n_blocks_local 9"

echo "=== V51 NIGHT armed $(date) ===" > "$LOG"
while ! grep -q 'DZ NIGHT BUILD DONE' "$CK/dz_night_log.txt" 2>/dev/null; do
  pgrep -f 'build_dz_night\.sh' > /dev/null || { echo "  DZ build gone -- abort" >> "$LOG"; exit 1; }
  sleep 120
done
# wait for the EVALUATION, not just the training -- eval_v50 renders 300 frames at loadSize
# 2048 and that render plus this training would exceed the card together
while ! grep -q 'EVAL V50 DONE' "$CK/eval_v50_log.txt" 2>/dev/null; do
  pgrep -f 'train_v50\.sh|eval_v50\.sh' > /dev/null || break
  sleep 120
done

# ---- assemble the corpus as symlinks; nothing is copied -----------------------------------
mkdir -p "$DST"
for s in $SUBS; do mkdir -p "$DST/$s"; done
# MUST be a file, not a heredoc: `conda run` does not forward stdin, so `$CE python3 - <<PY`
# executes nothing and exits 0. That silently emptied the Dark Zurich build earlier today.
$CE python3 -u $BASE/build_v51_corpus.py "$DST" "$SRC" "$DZ" \
  --subs "$(echo $SUBS | tr ' ' ',')" >> "$LOG" 2>&1

N=$(ls "$DST/train_img" | wc -l)
echo "  training on $N images (v47 trained on $(ls $SRC/train_img | wc -l))" >> "$LOG"
NV47=$(ls $SRC/train_img | wc -l)
# The entire premise of v51 is MORE data. If assembly produced a corpus no bigger than v47's,
# training would burn 5 GPU-hours to answer a question it cannot answer, and the null result
# would look like "more data does not help night" when nothing was actually added.
if [ "$N" -le "$NV47" ]; then
  echo "  ABORT: corpus is $N vs v47's $NV47 -- Dark Zurich added nothing. Not training." >> "$LOG"
  echo "=== V51 NIGHT TRAIN FAILED (empty corpus) $(date) ===" >> "$LOG"
  exit 1
fi
if [ "$N" -lt 5000 ]; then
  echo "  NOTE: corpus $N is smaller than hoped; v51 may differ little from v47." >> "$LOG"
fi

echo "=== train $NAME from $BASEM $(date) ===" >> "$LOG"
cd $BASE/pix2pixHD
ok=0
for attempt in 1 2 3; do
  $CE python3 -u train.py --name $NAME --dataroot "$DST" $ARCH \
    --num_D 3 --lambda_feat 25 --loadSize 2048 --fineSize 1024 \
    --resize_or_crop scale_width_and_crop --load_pretrain $CK/$BASEM \
    --niter 8 --niter_decay 4 --lr 0.0001 \
    --save_epoch_freq 2 --batchSize 1 --gpu_ids 0 \
    $( [ $attempt -gt 1 ] && echo --continue_train ) >> "$LOG" 2>&1 && { ok=1; break; }
  echo "--- attempt $attempt failed, retrying ---" >> "$LOG"; sleep 30
done
cd $BASE
# arch is unchanged from v47, so any shape-mismatch fallback means something is wrong
grep -q 'not initialized' "$LOG" && \
  echo "!!! load_network fell back to random init -- arch drifted from v47, result invalid !!!" >> "$LOG"
[ "$ok" = 1 ] && echo "=== V51 NIGHT TRAINED $(date) ===" >> "$LOG" \
              || echo "=== V51 NIGHT TRAIN FAILED $(date) ===" >> "$LOG"

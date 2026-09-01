#!/bin/bash
# v50 -- chroma grafted onto v45 instead of demolishing it.
#
# v49 tried this and lost 44% of car detail. The cause was not the chroma channel: pix2pixHD's
# load_network drops any tensor whose shape changed, so widening the input 70->73 threw away
# BOTH first convs entirely (343,392 params, 70 of 73 channels perfectly good v45 weights) and
# left them at random init. Two epochs at lr 5e-5 could not rebuild that.
#
# v50 starts from a checkpoint where v45's 70 channels are grafted in place and the 3 chroma
# columns are zero, so step 0 is bit-identical to v45 and training can only add. Same corpus as
# v49 -- the ONLY differences are the init and the schedule, so the comparison is clean.
set -u
. "$(dirname "${BASH_SOURCE[0]}")/config.sh"
BASE=$CARLA2REAL_ROOT
CE="conda run -n $CARLA2REAL_ENV"
CK=$BASE/pix2pixHD/checkpoints
IDST=$CARLA2REAL_DATA/training_v49_chroma
NAME=carla2real_semantic_v50_graft
LOG=$CK/v50_log.txt
ARCH="--label_nc 65 --no_instance --edge_input --depth_input --normal_input --chroma_input \
--netG local --ngf 32 --n_downsample_global 4 --n_local_enhancers 1 --n_blocks_local 9"

echo "=== V50 armed, waiting for the v49 sunny towns to release the GPU $(date) ===" > "$LOG"
while ! grep -q 'V49 SUNNY TOWNS DONE\|ALL TOWNS DONE' "$CK/v49_sunny_towns_log.txt" 2>/dev/null; do
  pgrep -f 'v49_sunny_towns\.sh' > /dev/null || { echo "  sunny job gone, proceeding $(date)" >> "$LOG"; break; }
  sleep 120
done

# the graft is the whole point -- if it is missing there is nothing to train
[ -s "$CK/$NAME/latest_net_G.pth" ] || { echo "NO GRAFT CHECKPOINT -- abort" >> "$LOG"; exit 1; }
# keep the pristine graft; training overwrites latest_net_* in place
INIT=$CK/${NAME}_init
[ -d "$INIT" ] || { mkdir -p "$INIT"; cp "$CK/$NAME"/latest_net_*.pth "$INIT/"; }

echo "=== [1] train $NAME from grafted v45 $(date) ===" >> "$LOG"
cd $BASE/pix2pixHD
ok=0
for attempt in 1 2 3; do
  $CE python3 -u train.py --name $NAME --dataroot "$IDST" $ARCH \
    --num_D 3 --lambda_feat 25 --loadSize 2048 --fineSize 1024 \
    --resize_or_crop scale_width_and_crop --load_pretrain "$INIT" \
    --niter 3 --niter_decay 1 --lr 0.0001 \
    --save_epoch_freq 1 --batchSize 1 --gpu_ids 0 \
    $( [ $attempt -gt 1 ] && echo --continue_train ) >> "$LOG" 2>&1 && { ok=1; break; }
  echo "--- attempt $attempt failed, retrying ---" >> "$LOG"; sleep 30
done
cd $BASE

# GATE: pix2pixHD announces a shape-mismatch fallback. If that fired, the graft did not take
# and this run is just v49 again with a longer schedule -- worthless, and worth saying loudly.
if grep -q 'not initialized' "$LOG"; then
  echo "!!! GRAFT DID NOT TAKE -- load_network fell back to random init. Result is invalid. !!!" >> "$LOG"
fi

[ "$ok" = 1 ] && echo "=== V50 TRAINED $(date) ===" >> "$LOG" \
              || echo "=== V50 TRAIN FAILED $(date) ===" >> "$LOG"

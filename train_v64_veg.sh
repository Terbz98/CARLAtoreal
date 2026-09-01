#!/bin/bash
# v64: the vegetation gain of v63 without the road it cost.
#
# WHAT v63 SETTLED. 20 epochs with --veg_weight 4.0 --far_boost 3.0 fixed the deficit it targeted:
#     far/near detail 0.83 -> 1.00, far detail +19.5%, and visibly -- palm fronds resolve as
#     fronds, and v50's magenta speckle in the crowns is gone.
# It also failed the tail gate: road high-pass 1.81x the parent, and 2.1x a REAL PHOTOGRAPH, which
# is the number that matters -- the parent was already slightly above both references, so this is
# invented grain on the surface lane geometry is read from. Rendering epochs 8, 12, 16 and 20 shows
# road at 1.88x / 1.87x / 2.89x / 1.81x: present from the start, not late drift. No epoch passes.
#
# WHY. VGGLoss with a weight map computes (w*d).sum()/w.sum() -- a WEIGHTED MEAN. Emphasis on
# vegetation is therefore dilution of everything else by 1/mean(w). With less perceptual pressure
# to match the real road, the GAN term (unweighted, and rewarded for plausible texture) decides it.
#
# THE CHANGE. --veg_extra keeps the original objective at full strength and ADDS a vegetation-only
# perceptual term on top, so vegetation gains gradient without the road losing any. One re-run, not
# a sweep: the mechanism is identified, this tests it.
#
# 12 epochs rather than 20 -- v63's own checkpoints show the vegetation gain is fully present by
# epoch 8 (near 811, far 723, the best absolute numbers of the run).
set -u
. "$(dirname "${BASH_SOURCE[0]}")/config.sh"
BASE=$CARLA2REAL_ROOT
CE="conda run -n carla_env"
CK=$BASE/pix2pixHD/checkpoints
LOG=$CK/v64_log.txt
NAME=carla2real_semantic_v64_veg
PARENT=carla2real_semantic_v50_graft
DST=$BASE/datasets/training_v49_chroma
PHS=test_Town10HD_sunny_inst_gt
ARCH="--label_nc 65 --no_instance --edge_input --depth_input --normal_input --chroma_input \
--netG local --ngf 32 --n_downsample_global 4 --n_local_enhancers 1 --n_blocks_local 9"
. $BASE/gpu_wait.sh
echo "=== V64 additive vegetation run $(date) ===" > "$LOG"
echo "corpus: $(ls $DST/train_img | wc -l) images" >> "$LOG"

wait_for_gpu
MARK=$(wc -l < "$LOG")
( cd $BASE/pix2pixHD && $CE python3 -u train.py --name "$NAME" --dataroot "$DST" $ARCH \
    --veg_extra 3.0 --far_boost 3.0 \
    --num_D 3 --lambda_feat 25 --loadSize 2048 --fineSize 1024 \
    --resize_or_crop scale_width_and_crop --load_pretrain "$CK/$PARENT" \
    --niter 8 --niter_decay 4 --lr 0.00005 \
    --save_epoch_freq 2 --batchSize 1 --gpu_ids 0 ) >> "$LOG" 2>&1
tail -n +$MARK "$LOG" | grep -q 'not initialized' && {
  echo "!! shape mismatch -- parent discarded" >> "$LOG"; exit 1; }

echo "=== render + measure $(date) ===" >> "$LOG"
wait_for_gpu
rm -rf "$BASE/pix2pixHD/results/$NAME/${PHS}_latest"
( cd $BASE/pix2pixHD && $CE python3 -u test.py --name "$NAME" \
    --dataroot $BASE/datasets/training_v12_mapillary $ARCH \
    --loadSize 2048 --resize_or_crop scale_width --phase "$PHS" --how_many 99999 \
    --which_epoch latest --gpu_ids 0 ) >> "$LOG" 2>&1

$CE python3 $BASE/veg_report.py "$NAME" >> "$LOG" 2>&1
$CE python3 $BASE/tail_check.py "$NAME" "$PHS" >> "$LOG" 2>&1
$CE python3 $BASE/road_sky_ceiling.py >> "$LOG" 2>&1
$CE python3 $BASE/epoch_sweep.py \
    "$BASE/pix2pixHD/results/$NAME/${PHS}_latest/images" >> "$LOG" 2>&1
$CE python3 -c "
import cv2, glob
fs=sorted(glob.glob('$BASE/pix2pixHD/results/$NAME/${PHS}_latest/images/*_synthesized_image.jpg'))
if len(fs)>450:
    cv2.imwrite('$BASE/check_v64.png', cv2.resize(cv2.imread(fs[450]),(1600,800),interpolation=cv2.INTER_AREA))
    print('  wrote check_v64.png')" >> "$LOG" 2>&1
echo "=== V64 DONE $(date) ===" >> "$LOG"

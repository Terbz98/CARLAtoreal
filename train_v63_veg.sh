#!/bin/bash
# v63: a long training run aimed at the one thing post-processing provably cannot fix --
# the generator under-detailing vegetation, and getting worse with distance.
#
# THE MEASURED PROBLEM (Town10HD, vegetation-labelled pixels, Laplacian variance):
#     real training photos   near 1191   far 1246   far/near 1.05
#     CARLA source           near  928   far 1030   far/near 1.11
#     v50 render             near  672   far  561   far/near 0.83
# Two deficits. The model reaches 56% of the detail present in the photographs it was trained on,
# and where both references HOLD detail at distance, the render loses it. Distant trees are
# therefore doubly penalised, which is exactly what is reported.
#
# WHY IT HAPPENS. L1 and the VGG perceptual loss are area-weighted. A distant tree covers a few
# hundred pixels out of two million, contributes a proportionate share of the gradient, and is
# never learned. Nothing in the objective says small-and-far matters. This is the same failure
# that lost the motorcycle at 54% detection before --thing_weight existed.
#
# THE FIX. --veg_weight upweights vegetation in the perceptual loss; --far_boost scales that
# weight by distance using the inverse-depth channel the model already receives (verified: sky
# reads 0.0, road ~170, so FAR is the low end). Vegetation far away therefore carries gradient in
# proportion to how badly it is currently rendered rather than to its pixel count.
#
# WHY THIS AND NOT MORE FILTERING. Every post-processing attempt traded detail for stability,
# because a filter can only remove. Three days of GPU is the right instrument for a deficit that
# is about what the model never learned.
#
# Corpus is the full 32k mixed set v50 was trained on -- NOT the 9k sequence corpus -- because
# this is about image quality, not temporal behaviour, and diversity is what protects against
# drifting off Mapillary's detail.
set -u
. "$(dirname "${BASH_SOURCE[0]}")/config.sh"
BASE=$CARLA2REAL_ROOT
CE="conda run -n carla_env"
CK=$BASE/pix2pixHD/checkpoints
LOG=$CK/v63_log.txt
NAME=carla2real_semantic_v63_veg
PARENT=carla2real_semantic_v50_graft
DST=$BASE/datasets/training_v49_chroma
PHS=test_Town10HD_sunny_inst_gt
ARCH="--label_nc 65 --no_instance --edge_input --depth_input --normal_input --chroma_input \
--netG local --ngf 32 --n_downsample_global 4 --n_local_enhancers 1 --n_blocks_local 9"
. $BASE/gpu_wait.sh
echo "=== V63 vegetation run $(date) ===" > "$LOG"
echo "corpus: $(ls $DST/train_img | wc -l) images" >> "$LOG"

# No graft needed: the architecture is unchanged, only the loss weighting. --load_pretrain
# therefore restores v50 exactly, and any 'not initialized' would mean something is wrong.
wait_for_gpu
MARK=$(wc -l < "$LOG")
( cd $BASE/pix2pixHD && $CE python3 -u train.py --name "$NAME" --dataroot "$DST" $ARCH \
    --veg_weight 4.0 --far_boost 3.0 \
    --num_D 3 --lambda_feat 25 --loadSize 2048 --fineSize 1024 \
    --resize_or_crop scale_width_and_crop --load_pretrain "$CK/$PARENT" \
    --niter 14 --niter_decay 6 --lr 0.00005 \
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
$CE python3 -c "
import cv2, glob
fs=sorted(glob.glob('$BASE/pix2pixHD/results/$NAME/${PHS}_latest/images/*_synthesized_image.jpg'))
if len(fs)>450:
    cv2.imwrite('$BASE/check_v63.png', cv2.resize(cv2.imread(fs[450]),(1600,800),interpolation=cv2.INTER_AREA))
    print('  wrote check_v63.png')" >> "$LOG" 2>&1
echo "=== V63 DONE $(date) ===" >> "$LOG"

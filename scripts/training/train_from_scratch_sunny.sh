#!/bin/bash
# v91c -- the first sunny model whose weights have never seen restricted data.
#
# WHY FROM SCRATCH. Every sunny baseline since v6 has been FINE-TUNED from the one before. v90's final
# training data is PandaSet + ZOD only, but its weights descend v90 <- v75 <- v73 <- v50, and those
# trained on Mapillary Vistas (non-commercial), Cityscapes (research only) and the 21 videos of
# unestablished provenance. The night line is the same: v79 <- v76 <- v69 <- v51 <- v47, through
# Dark Zurich and the unlicensed night videos. If trained weights are a derivative of their training
# data -- which is the repository's own stated reason for licensing them CC BY-SA -- then every
# published checkpoint so far inherits those restrictions. The only clean answer is random init on
# licensed data. Found 2026-10-01; the user chose this over more fine-tuning.
#
# COARSE TO FINE, as pix2pixHD was designed to be trained from scratch:
#   stage 1  the global generator alone (ngf 64), at half resolution (loadSize 1024, crop 512)
#   stage 2  the 2048 local enhancer wrapped around it (ngf 32 -> global ngf 64, the same layers),
#            outer layers alone for NITER_FIX epochs, then jointly
# The architecture of stage 2 is exactly v90's, so render_model.sh, the delivery chain, the vehicle
# pass and the aerial pass all apply unchanged.
#
# THE VOID-RUN CHECK HAS TO BE DIFFERENT HERE. Loading a global checkpoint into a local enhancer
# MUST report the new outer layers (model1_1, model1_2) as "not initialized" -- that is the design.
# What must never appear is 'model' itself, the global generator: that is the v86 failure, and it
# would mean stage 2 restarted from noise. So the check is on the exact list, not the phrase.
#
# Corpus: training_v90 -- PandaSet 19,158 (three forward cameras, daylight, colour-matched) + ZOD
# 10,000. Every frame CC BY 4.0 or CC BY-SA 4.0.
set -u
BASE=${CARLA2REAL_ROOT:?set CARLA2REAL_ROOT (see README)}
PY=${PY:-python3}
CK=$BASE/pix2pixHD/checkpoints
DATA=${CARLA2REAL_DATA:?set CARLA2REAL_DATA}/training_v90
LOG=$CK/v91c_log.txt
G1=carla2real_semantic_v91c_global
G2=carla2real_semantic_v91c
IN="--label_nc 65 --no_instance --edge_input --depth_input --normal_input --chroma_input --texture_input"
S1_NITER=${S1_NITER:-12}; S1_DECAY=${S1_DECAY:-8}
S2_NITER=${S2_NITER:-6};  S2_DECAY=${S2_DECAY:-4}; NITER_FIX=${NITER_FIX:-2}
cd "$BASE"
say () { echo "$(date '+%m-%d %H:%M')  $*" >> "$LOG"; }

say "=== v91c from-scratch licensed retrain (pid $$) ==="
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 120; done

N=$(ls "$DATA/train_img" | wc -l)
for d in train_label train_edge train_depth train_normal train_chroma train_texture; do
  [ "$(ls "$DATA/$d" | wc -l)" -eq "$N" ] || { say "ABORT: $d incomplete -- stopping"; exit 1; }
done
say "corpus $N pairs: $(ls "$DATA/train_img" | sed 's/[0-9]*\..*//' | sort | uniq -c | tr -s ' ' | tr '\n' ' ')"

# ---- smoke test: both stages on 20 frames, ~5 minutes ----------------------------------------
# The GPU was busy when this was written, so neither stage could be tried. A wrong argument in stage 2
# would otherwise surface only after thirteen hours of stage 1, with the GPU then idle until someone
# looked. Same arguments as the real run, tiny dataset, one epoch each, including the load check.
if ! grep -q 'SMOKE PASSED' "$LOG"; then
  say "smoke test"
  rm -rf "$CK/${G1}_smoke" "$CK/${G2}_smoke"
  ( cd pix2pixHD && $PY -u train.py --name ${G1}_smoke --dataroot "$DATA" $IN \
      --netG global --ngf 64 --n_downsample_global 4 --n_blocks_global 9 \
      --num_D 3 --lambda_feat 25 --loadSize 1024 --fineSize 512 --resize_or_crop scale_width_and_crop \
      --niter 1 --niter_decay 0 --lr 0.0002 --save_epoch_freq 1 --batchSize 1 --gpu_ids 0 --max_dataset_size 20 ) > "$CK/v91c_smoke1.txt" 2>&1
  [ -s "$CK/${G1}_smoke/latest_net_G.pth" ] || { say "SMOKE stage 1 FAILED -- see v91c_smoke1.txt"; exit 1; }
  ( cd pix2pixHD && $PY -u train.py --name ${G2}_smoke --dataroot "$DATA" $IN \
      --netG local --ngf 32 --n_downsample_global 4 --n_local_enhancers 1 --n_blocks_local 9 \
      --num_D 3 --lambda_feat 25 --loadSize 2048 --fineSize 1024 --resize_or_crop scale_width_and_crop \
      --niter 1 --niter_decay 0 --niter_fix_global 1 --lr 0.0001 --save_epoch_freq 1 --batchSize 1 --gpu_ids 0 \
      --max_dataset_size 20 --load_pretrain "$CK/${G1}_smoke" ) > "$CK/v91c_smoke2.txt" 2>&1
  [ -s "$CK/${G2}_smoke/latest_net_G.pth" ] || { say "SMOKE stage 2 FAILED -- see v91c_smoke2.txt"; exit 1; }
  M=$(grep -A1 'Pretrained network G has fewer layers' "$CK/v91c_smoke2.txt" | tail -1)
  say "smoke stage 2 G uninitialised: ${M:-none}"
  echo "$M" | grep -qE "'model'" && { say "SMOKE: global generator not loaded into stage 2 -- stopping"; exit 1; }
  rm -rf "$CK/${G1}_smoke" "$CK/${G2}_smoke"
  say "SMOKE PASSED"
fi

# ---- stage 1: global generator from random init -------------------------------------------------
if [ ! -s "$CK/$G1/latest_net_G.pth" ] || ! grep -q 'STAGE1 DONE' "$LOG"; then
  say "stage 1: global, ${S1_NITER}+${S1_DECAY} epochs at 1024/512, lr 2e-4, random init"
  ( cd pix2pixHD && $PY -u train.py --name $G1 --dataroot "$DATA" $IN \
      --netG global --ngf 64 --n_downsample_global 4 --n_blocks_global 9 \
      --num_D 3 --lambda_feat 25 --loadSize 1024 --fineSize 512 --resize_or_crop scale_width_and_crop \
      --niter $S1_NITER --niter_decay $S1_DECAY --lr 0.0002 --save_epoch_freq 1 --batchSize 1 --gpu_ids 0 \
      $( [ -s "$CK/$G1/latest_net_G.pth" ] && echo --continue_train ) ) > "$CK/${G1}_log.txt" 2>&1
  [ -s "$CK/$G1/latest_net_G.pth" ] || { say "STAGE 1 FAILED -- no checkpoint, stopping"; exit 1; }
  say "STAGE1 DONE"
fi

# ---- stage 2: local enhancer around the trained global ------------------------------------------
if ! grep -q 'STAGE2 DONE' "$LOG"; then
  say "stage 2: local enhancer, ${S2_NITER}+${S2_DECAY} epochs (outer layers alone for $NITER_FIX), lr 1e-4"
  RESUME=""; INIT="--load_pretrain $CK/$G1"
  [ -s "$CK/$G2/latest_net_G.pth" ] && { RESUME="--continue_train"; INIT=""; }
  ( cd pix2pixHD && $PY -u train.py --name $G2 --dataroot "$DATA" $IN \
      --netG local --ngf 32 --n_downsample_global 4 --n_local_enhancers 1 --n_blocks_local 9 \
      --num_D 3 --lambda_feat 25 --loadSize 2048 --fineSize 1024 --resize_or_crop scale_width_and_crop \
      --niter $S2_NITER --niter_decay $S2_DECAY --niter_fix_global $NITER_FIX --lr 0.0001 \
      --save_epoch_freq 1 --batchSize 1 --gpu_ids 0 $INIT $RESUME ) > "$CK/${G2}_log.txt" 2>&1 &
  TP=$!
  # check the init within the first minutes, not after seventeen hours
  for i in $(seq 1 60); do
    grep -q 'iters: 100,' "$CK/${G2}_log.txt" 2>/dev/null && break
    kill -0 $TP 2>/dev/null || break; sleep 10
  done
  if [ -z "$RESUME" ]; then
    MISSING=$(grep -A1 'Pretrained network G has fewer layers' "$CK/${G2}_log.txt" | tail -1)
    say "stage 2 G uninitialised list: ${MISSING:-none}"
    if echo "$MISSING" | grep -qE "'model'"; then
      kill $TP; say "VOID: the global generator was NOT loaded into stage 2 -- killed. This is the v86 trap."; exit 1
    fi
  fi
  wait $TP
  [ -s "$CK/$G2/latest_net_G.pth" ] || { say "STAGE 2 FAILED -- no checkpoint, stopping"; exit 1; }
  say "STAGE2 DONE"
fi

say "=== trained: $CK/$G2 -- render with scripts/inference/render_model.sh ==="

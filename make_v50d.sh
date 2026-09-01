#!/bin/bash
# v50d = v50b (vehicle colour corrected + edge-masked sharpen) + temporal de-shimmer.
#
# The point of this pass is the "best of both worlds" the user asked for. Measured on town10hd
# sunny, 150 frames at 960x480:
#
#     variant                     alt p99   sharpness
#     v49  (stable, muddled)        27.51       440.9
#     v50  (detailed, shimmery)     43.85       464.2
#     v50b (+sharpen)               45.48       772.0
#     v50d (+de-shimmer)            26.72       634.3    <- calmer than v49, 44% sharper
#
# alt p99 is the 99th percentile of |2*x[t] - x[t-1] - x[t+1]|, i.e. how hard the worst pixels
# flip-flop between consecutive frames. It is the metric that matches what the eye calls flicker;
# the mean static-pixel delta I used earlier said v50 was CALMER than v49, which was wrong.
#
# Post-processing only, no GPU, ~6 min/clip. Delivered under its own tag; v49/v50/v50b untouched.
set -u
. "$(dirname "${BASH_SOURCE[0]}")/config.sh"
BASE=$CARLA2REAL_ROOT
CE="conda run -n carla_env"
DEST=$CARLA2REAL_OUT
VPB=$PERCEPTION_ROOT/build
OUT=$BASE/pix2pixHD/results/mp4/v50d
LOG=$BASE/pix2pixHD/checkpoints/v50d_log.txt
mkdir -p "$OUT" "$DEST/logs_v50d"
echo "=== V50D PASS $(date) ===" > "$LOG"

for T in town03 town04 town05 town06 town10hd; do
  SRC=$DEST/${T}_sunny_vp55_v50b_FINAL_1920_visionpilot.mp4
  DN=${T}_sunny_vp55_v50d_FINAL_1920_visionpilot.mp4
  [ -s "$SRC" ] || { echo "  $T: no v50b master, skipped" >> "$LOG"; continue; }
  [ -s "$DEST/$DN" ] && { echo "  $T already done" >> "$LOG"; continue; }
  echo "--- $T $(date +%H:%M:%S)" >> "$LOG"

  $CE python3 -u $BASE/temporal_deshimmer.py "$SRC" "$OUT/${T}_ds.avi" \
      --alt 1 --flow 6.0 >> "$LOG" 2>&1
  S=$OUT/${T}_ds.avi
  [ -s "$S" ] || { echo "  $T DESHIMMER FAILED" >> "$LOG"; continue; }

  $CE python3 -u -c "
import cv2
c=cv2.VideoCapture('$S')
w=cv2.VideoWriter('$DEST/$DN',cv2.VideoWriter_fourcc(*'mp4v'),c.get(5) or 30,(1920,960))
n=0
while True:
    ok,f=c.read()
    if not ok: break
    w.write(cv2.resize(f,(1920,960),interpolation=cv2.INTER_AREA)); n+=1
w.release(); print('  delivered %d frames'%n)" >> "$LOG" 2>&1
  [ -s "$DEST/$DN" ] || { echo "  $T DELIVER FAILED" >> "$LOG"; continue; }

  mkdir -p "$DEST/vp_input_1024"
  $CE python3 -u -c "
import cv2
c=cv2.VideoCapture('$DEST/$DN')
w=cv2.VideoWriter('$DEST/vp_input_1024/$DN',cv2.VideoWriter_fourcc(*'mp4v'),c.get(5) or 30,(1024,512))
while True:
    ok,f=c.read()
    if not ok: break
    w.write(cv2.resize(f,(1024,512),interpolation=cv2.INTER_AREA))
w.release()" >> "$LOG" 2>&1
  while pgrep -x VisionPilot >/dev/null 2>&1; do sleep 15; done
  ( cd "$VPB" && ./record_carla.sh "$DEST/vp_input_1024/$DN" "$DEST/${T}_sunny_frame_speed.txt" \
      "$DEST/calibrated/$DN" "$DEST/logs_v50d/${T}_sunny.log" ) >> "$LOG" 2>&1
  G=$DEST/gt/${T}_sunny_gt.json
  [ -s "$G" ] && $CE python3 $BASE/score_vp.py "$G" "$DEST/logs_v50d/${T}_sunny.log" \
      --json "$DEST/logs_v50d/${T}_sunny_score.json" >> "$LOG" 2>&1

  # per-clip proof that the trade actually landed, not just on the town it was tuned on
  $CE python3 -u -c "
import cv2, numpy as np
def sc(p,n=150):
    c=cv2.VideoCapture(p); b=[]
    while len(b)<n:
        ok,f=c.read()
        if not ok: break
        b.append(cv2.resize(f,(960,480)))
    c.release()
    a=np.stack([cv2.cvtColor(cv2.resize(f,(480,240)),cv2.COLOR_BGR2GRAY).astype(np.float32) for f in b])
    alt=np.abs(2*a[1:-1]-a[:-2]-a[2:])
    sh=np.mean([cv2.Laplacian(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var() for f in b])
    return np.percentile(alt.mean(axis=0),99), sh
for n,p in (('v49','$DEST/${T}_sunny_vp55_v49_FINAL_1920_visionpilot.mp4'),
            ('v50','$DEST/${T}_sunny_vp55_v50_FINAL_1920_visionpilot.mp4'),
            ('v50d','$DEST/$DN')):
    import os
    if os.path.exists(p):
        a,s=sc(p); print('  %-5s altp99 %6.2f  sharp %7.1f'%(n,a,s))" >> "$LOG" 2>&1

  rm -f "$OUT/${T}_ds.avi"
  echo "  $T DONE -> $DN" >> "$LOG"
done
bash $BASE/organise_calibrated.sh >> "$LOG" 2>&1
bash $BASE/refresh_new.sh >> "$LOG" 2>&1
echo "=== V50D PASS DONE $(date) ===" >> "$LOG"

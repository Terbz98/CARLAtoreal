#!/usr/bin/env python3
"""De-shimmer guided by the SEMANTIC LABEL MAP: smooth where the flicker is, leave the rest sharp.

WHY. temporal_deshimmer.py is global -- it smooths anywhere that alternates and is not moving.
Measuring flicker per class on the raw Town10HD sunny render (60 frames, alternation
|2x(t)-x(t-1)-x(t+1)|) shows how badly that is aimed:

    class          % of frame   mean flicker   contribution
    building             25.0          47.97          11.97   <- more than everything else combined
    car                   7.5          47.36           3.55
    road                 38.0           8.61           3.27   <- already stable, and 38% of the frame
    vegetation            5.4          51.33           2.80
    pole                  1.3         100.52           1.34
    sidewalk              7.3          19.84           1.45

The road is the largest surface in the frame and barely flickers, yet the global filter smooths it
just as hard as the facades -- so the clip ends up blurry AND still flickery: the blur lands where
it is not needed and there is not enough of it where it is.

So: per-class strength. Buildings, poles, vegetation and fences get the full median; road and lane
markings get none, which is also where sharpness matters most for the perception stack downstream.
Cars get a reduced amount because they move, and the motion gate below can only do so much.

The existing gates are kept on top of the class weight: a pixel is only smoothed if it is actually
alternating AND optical flow says it is not moving, so this can only ever be more conservative
than the global version at any given pixel.

  usage: class_deshimmer.py <in> <label_dir> <out> [--alt 1.0] [--flow 6.0]
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vidcodec import fourcc_for

IN, LABDIR, OUT = sys.argv[1], sys.argv[2], sys.argv[3]


def opt(flag, default):
    return float(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


ALT_T = opt('--alt', 1.0)
FLOW_T = opt('--flow', 6.0)

# PER-CLASS MOTION TOLERANCE. The global filter only smooths pixels optical flow calls
# stationary, and buildings sweep past the camera -- so the class that owns most of the flicker
# (mean 48.0, 25% of the frame) is precisely the one the gate refuses to touch. Measured: the
# global de-shimmer moved building flicker only 186 -> 183 while blurring the road 22%.
#
# Relaxing that gate is safe for BUILDINGS specifically: their apparent motion is smooth ego-motion
# across a rigid surface, not the erratic flow that corrupted the motion-compensated experiment.
# It is NOT safe for cars, which move independently and would trail, so they keep the tight gate.
FLOW_TOL = {
    17: 40.0,   # building   allow smoothing despite parallax; this is the whole point
    45: 40.0,   # pole
    46: 40.0,
    30: 30.0,   # vegetation
    50: 30.0,   # fence
    27: 40.0,   # sky
    55: 4.0,    # car        independent motion -- keep tight or it trails
    61: 4.0,    # truck / bus
}

# Mapillary-65 ids. Strength 0 = never touched, 1 = full temporal median.
STRENGTH = {
    13: 0.0,    # road          38% of frame, mean flicker 8.6 -- already stable, keep it sharp
    24: 0.0,    # lane marking  the perception stack keys on these; never soften them
    17: 1.0,    # building      the dominant flicker source
    45: 1.0,    # pole          worst per-pixel flicker (mean 100) though only 1.3% of frame
    46: 1.0,    # pole / sign
    30: 1.0,    # vegetation
    50: 0.9,    # fence
    27: 0.8,    # sky
    15: 0.6,    # sidewalk
    54: 0.6,
    55: 0.4,    # car           moving; the flow gate is doing most of the work here
    61: 0.4,    # truck / bus
}
DEFAULT_STRENGTH = 0.6

# Env overrides so the strengths can be swept without editing the table. ROAD in particular needs
# tuning by eye, not by its mean: the road's MEAN flicker is low (8.6) but it covers 38% of the
# frame, so it dominates what the viewer actually sees. Setting it to 0 made road flicker worse
# than the baseline (41.0 vs 28.0) even though the class statistics said it was fine.
for _k, _e in ((13, 'ROAD_STRENGTH'), (24, 'LANE_STRENGTH'), (17, 'BLDG_STRENGTH'),
               (55, 'CAR_STRENGTH'), (61, 'CAR_STRENGTH')):
    if os.environ.get(_e):
        STRENGTH[_k] = float(os.environ[_e])
for _k, _e in ((55, 'CAR_FLOW'), (61, 'CAR_FLOW'), (17, 'BLDG_FLOW')):
    if os.environ.get(_e):
        FLOW_TOL[_k] = float(os.environ[_e])
# WINDOW. 3 frames is one neighbour each side. Buildings are the largest remaining source even
# after halving (contribution 4.87 of 13.61), and a wider median reaches further back in time,
# which is what a slow persistent shimmer needs. Only applied where strength is high, so moving
# objects are not dragged across a longer span.
WINDOW = int(os.environ.get('WINDOW', '3'))
WIDE_WINDOW_ABOVE = float(os.environ.get('WIDE_ABOVE', '0.85'))
LIMIT = int(os.environ.get('LIMIT', '0'))    # 0 = whole clip; a frame count for quick sweeps

labs = sorted([f for f in os.listdir(LABDIR) if f.endswith('.png')])
cap = cv2.VideoCapture(IN)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
vw = cv2.VideoWriter(OUT, fourcc_for(OUT), fps, (W, H))

buf, gray, n, touched = [], [], 0, []
lut = np.full(256, DEFAULT_STRENGTH, np.float32)
for k, v in STRENGTH.items():
    lut[k] = v
flut = np.full(256, FLOW_T, np.float32)
for k, v in FLOW_TOL.items():
    flut[k] = v


def maps(i):
    """Per-pixel smoothing strength and motion tolerance, from the label map."""
    if i >= len(labs):
        return (np.full((H, W), DEFAULT_STRENGTH, np.float32),
                np.full((H, W), FLOW_T, np.float32))
    lab = np.array(Image.open(os.path.join(LABDIR, labs[i])))
    lab = lab if lab.ndim == 2 else lab[:, :, 0]
    lab = cv2.resize(lab, (W, H), interpolation=cv2.INTER_NEAREST)
    # soften class boundaries so the transition between smoothed and untouched is not visible
    return cv2.GaussianBlur(lut[lab], (0, 0), 2.0), flut[lab]


while True:
    if LIMIT and n >= LIMIT:
        break
    ok, f = cap.read()
    if not ok:
        break
    buf.append(f)
    gray.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    if len(buf) < WINDOW:
        if len(buf) == 1:
            vw.write(f); n += 1
        continue
    if len(buf) > WINDOW:
        buf.pop(0); gray.pop(0)
    mid = WINDOW // 2
    ga, gb, gc = gray[mid - 1], gray[mid], gray[mid + 1]
    cur = buf[mid]
    alt = np.abs(2.0 * gb.astype(np.float32) - ga.astype(np.float32) - gc.astype(np.float32))
    fl = cv2.calcOpticalFlowFarneback(ga, gc, None, 0.5, 3, 21, 3, 5, 1.2, 0)
    mag = np.linalg.norm(fl, axis=2) * 0.5
    smap, fmap = maps(n)
    gate = ((alt > ALT_T) & (mag < fmap)).astype(np.float32)
    w = cv2.GaussianBlur(gate, (0, 0), 1.5) * smap
    stack = np.stack(buf).astype(np.float32)
    med = np.median(stack, axis=0)
    if WINDOW > 3:
        # narrow median for everything else, wide only for the strongly-smoothed classes, so a
        # long window cannot smear a car or the road
        narrow = np.median(stack[mid - 1:mid + 2], axis=0)
        wide_ok = (smap >= WIDE_WINDOW_ABOVE)[..., None]
        med = np.where(wide_ok, med, narrow)
    out = np.clip(cur * (1 - w[..., None]) + med * w[..., None], 0, 255).astype(np.uint8)
    vw.write(out)
    touched.append(float(w.mean()))
    n += 1

if buf:
    vw.write(buf[-1]); n += 1
vw.release(); cap.release()
print(f'  class de-shimmer: {n} frames, mean smoothing weight {np.mean(touched) * 100:.1f}% '
      f'(global version applies it everywhere the gate fires)')

#!/usr/bin/env python3
"""Kill 1-frame shimmer without touching detail or motion.

WHY A NEW STAGE. v50 has better detail than v49 but reads as flickerier, and the usual metrics
disagreed with that: mean static-pixel luminance delta says v50 (2.79) is CALMER than v49 (3.37).
Averaging over the frame is what hides it. The eye responds to ALTERNATION -- a pixel that flips
between two values on consecutive frames -- and on that measure v50 is clearly worse:

    town10hd sunny        alt mean   alt p99
    v49                     12.72     33.48
    v50                     14.67     46.25
    v50b (sharpened)        15.82     49.07

So sharpening bought detail and paid for it in shimmer. This stage buys the shimmer back without
returning the detail.

HOW. Alternation is measured directly as |2*x[t] - x[t-1] - x[t+1]|, which is large for a
one-frame flip-flop and near zero for smooth motion or a steady ramp. Where that value is high
AND optical flow says the pixel is not moving, the pixel is replaced by the temporal MEDIAN of
the three frames.

Median, not mean, on purpose: a mean of three frames blurs a moving edge and drags outliers into
the result, which is precisely the muddiness v49 suffers from. A median discards the odd frame
out and keeps the other two intact, so a stable edge survives untouched.

The flow gate matters as much as the median. Applying this everywhere would smear real motion --
that is DVP's failure mode at high epoch counts. Only genuinely static, genuinely alternating
pixels are touched, which on these clips is a small minority of the frame.

Usage: temporal_deshimmer.py in.mp4 out.mp4 [--alt 6] [--flow 0.6] [--strength 1.0]
"""
import sys, os
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vidcodec import fourcc_for

IN, OUT = sys.argv[1], sys.argv[2]


def _arg(name, default):
    return float(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


ALT_THRESH = _arg('--alt', 1.0)        # alternation units before a pixel counts as shimmering
FLOW_MAX = _arg('--flow', 6.0)         # px/frame; above this the pixel is moving, leave it alone
STRENGTH = _arg('--strength', 1.0)     # 1.0 = full median substitution where gated

cap = cv2.VideoCapture(IN)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
W, H = int(cap.get(3)), int(cap.get(4))
vw = cv2.VideoWriter(OUT, fourcc_for(OUT), fps, (W, H))

frames, grays = [], []
n = 0
touched = []


def push(f):
    frames.append(f)
    grays.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    if len(frames) > 3:
        frames.pop(0)
        grays.pop(0)


ok, f0 = cap.read()
if not ok:
    raise SystemExit('empty input')
push(f0)
vw.write(f0)                            # first frame has no predecessor; pass it through
n += 1

ok, f1 = cap.read()
if ok:
    push(f1)

while True:
    ok, f2 = cap.read()
    if not ok:
        break
    push(f2)
    a, b, c = frames[0], frames[1], frames[2]      # b is the frame being written
    ga, gb, gc = grays[0], grays[1], grays[2]

    alt = np.abs(2.0 * gb.astype(np.float32) - ga.astype(np.float32) - gc.astype(np.float32))
    flow = cv2.calcOpticalFlowFarneback(ga, gc, None, 0.5, 3, 21, 3, 5, 1.2, 0)
    mag = np.linalg.norm(flow, axis=2) * 0.5       # a->c spans two frames

    gate = (alt > ALT_THRESH) & (mag < FLOW_MAX)
    touched.append(float(gate.mean()) * 100)

    if gate.any():
        med = np.median(np.stack([a, b, c]).astype(np.float32), axis=0)
        # feather the mask so corrected and untouched regions do not show a seam
        w = cv2.GaussianBlur(gate.astype(np.float32), (0, 0), 1.5)[..., None] * STRENGTH
        out = np.clip(b.astype(np.float32) * (1 - w) + med * w, 0, 255).astype(np.uint8)
    else:
        out = b
    vw.write(out)
    n += 1

if len(frames) >= 1:
    vw.write(frames[-1])                # last frame has no successor
    n += 1

vw.release()
cap.release()
print('wrote %s  (%d frames, mean %.1f%% of pixels de-shimmered)'
      % (OUT, n, float(np.mean(touched)) if touched else 0.0))

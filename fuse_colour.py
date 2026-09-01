#!/usr/bin/env python3
"""Fuse v50d's stability with v63's colour: transfer the colour, leave the structure alone.

THE TWO CLIPS, measured on Town10HD frames 300-360:
    clip    flicker (alt p99)   saturation   colourfulness   colour drift/frame
    v50d               160.00         48.9            18.8                0.320
    v63                189.00         44.1            23.1                0.637
v63 is 18% flickerier and 23% more colourful. Note that its SATURATION is lower: the vibrancy is
not saturation, it is colour SEPARATION -- v50d pushes the whole frame toward one cast, so its cars
read muddy brown where v63's red reads red.

WHY A STRAIGHT COPY WOULD FAIL. That last column is the problem. v63's own frame-to-frame colour
drift is twice v50d's, so matching colour per frame would import exactly the flicker being avoided.
So the transfer statistics are SMOOTHED over a temporal window first: only the slow colour trend
crosses over, never the jitter. v50d stays the carrier of every edge, every pixel of motion, and all
of its stability.

Both clips are rendered from the SAME recording -- identical camera path, identical traffic -- so
frame N of one corresponds exactly to frame N of the other and no alignment is needed.

L is transferred at a lower gain than a and b by default: shifting luminance changes exposure and
with it the apparent sharpness, which is not what is being borrowed here.

  usage: fuse_colour.py <carrier.mp4> <colour_source.mp4> <out.avi> [strength] [window]
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vidcodec import fourcc_for

CARRIER, SOURCE, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
STRENGTH = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0     # 0 = untouched v50d, 1 = full v63 colour
WINDOW = int(sys.argv[5]) if len(sys.argv) > 5 else 61          # frames; must be odd
L_GAIN = float(os.environ.get('L_GAIN', 0.35))                  # luminance moves less than chroma
MAX_STD_RATIO = 1.6                                             # never stretch contrast wildly


def stats(path):
    """Per-frame Lab mean and std."""
    c = cv2.VideoCapture(path)
    m, s = [], []
    while True:
        ok, f = c.read()
        if not ok:
            break
        lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32)
        m.append(lab.reshape(-1, 3).mean(0))
        s.append(lab.reshape(-1, 3).std(0))
    c.release()
    return np.array(m), np.array(s)


def smooth(a, w):
    """Moving average along time, edge-padded. This is what strips v63's colour jitter."""
    if w <= 1 or len(a) < 3:
        return a
    w = min(w | 1, (len(a) // 2) * 2 - 1)
    if w < 3:
        return a
    pad = w // 2
    p = np.pad(a, ((pad, pad), (0, 0)), mode='edge')
    k = np.ones(w) / w
    return np.stack([np.convolve(p[:, i], k, mode='valid') for i in range(a.shape[1])], axis=1)


mc, sc = stats(CARRIER)
ms, ss = stats(SOURCE)
n = min(len(mc), len(ms))
if n == 0:
    raise SystemExit('one of the clips has no frames')
mc, sc, ms, ss = mc[:n], sc[:n], ms[:n], ss[:n]
mcs, scs, mss, sss = smooth(mc, WINDOW), smooth(sc, WINDOW), smooth(ms, WINDOW), smooth(ss, WINDOW)

gain = np.clip(sss / np.maximum(scs, 1e-3), 1.0 / MAX_STD_RATIO, MAX_STD_RATIO)
chan_w = np.array([L_GAIN, 1.0, 1.0]) * STRENGTH
gain = 1.0 + (gain - 1.0) * chan_w
shift = (mss - mcs) * chan_w

c = cv2.VideoCapture(CARRIER)
fps = c.get(cv2.CAP_PROP_FPS) or 30
W, H = int(c.get(3)), int(c.get(4))
w = cv2.VideoWriter(OUT, fourcc_for(OUT), fps, (W, H))
for i in range(n):
    ok, f = c.read()
    if not ok:
        break
    lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32)
    for ch in range(3):
        lab[:, :, ch] = (lab[:, :, ch] - mcs[i, ch]) * gain[i, ch] + mcs[i, ch] + shift[i, ch]
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    w.write(out)
c.release()
w.release()
print(f'  wrote {OUT}  ({n} frames, strength {STRENGTH}, window {WINDOW}, L gain {L_GAIN})')
print(f'  mean Lab shift applied: L {shift[:,0].mean():+.2f}  a {shift[:,1].mean():+.2f}  b {shift[:,2].mean():+.2f}')

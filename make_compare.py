#!/usr/bin/env python3
"""Build a side-by-side comparison mp4 from two delivered clips.

Both sources are 1920x960. Placed side by side at full size that would be 3840 wide, which is
awkward to view, so each is halved to 960x480 and the pair sits in a 1920x480 frame with a divider
and a caption bar. Frame N of the left is shown against frame N of the right, which is meaningful
here because both clips are rendered from the SAME recording — identical camera path, identical
traffic — so any difference on screen is the model, not the drive.

  usage: make_compare.py <left.mp4> <right.mp4> <out.mp4> <left label> <right label>
"""
import sys

import cv2
import numpy as np

L, R, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
LAB_L = sys.argv[4] if len(sys.argv) > 4 else 'left'
LAB_R = sys.argv[5] if len(sys.argv) > 5 else 'right'

cl, cr = cv2.VideoCapture(L), cv2.VideoCapture(R)
fps = cl.get(cv2.CAP_PROP_FPS) or 30
W, H = 960, 480
BAR = 42
out = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W * 2, H + BAR))

n = 0
while True:
    okl, fl = cl.read()
    okr, fr = cr.read()
    if not (okl and okr):
        break
    fl = cv2.resize(fl, (W, H))
    fr = cv2.resize(fr, (W, H))
    canvas = np.zeros((H + BAR, W * 2, 3), np.uint8)
    canvas[BAR:, :W] = fl
    canvas[BAR:, W:] = fr
    # caption bar, and a divider so the join is unambiguous on a dark night scene
    cv2.putText(canvas, LAB_L, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(canvas, LAB_R, (W + 14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(canvas, f'frame {n}', (W * 2 - 190, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (160, 160, 160), 2)
    cv2.line(canvas, (W, BAR), (W, H + BAR), (90, 90, 90), 2)
    out.write(canvas)
    n += 1

out.release(); cl.release(); cr.release()
print(f'  wrote {OUT}  ({n} frames, {W*2}x{H+BAR})')

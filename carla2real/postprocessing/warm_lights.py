#!/usr/bin/env python3
"""Subtle warm (sodium-ish) tint on night light sources and their glow.

WHY. User, 2026-10-08, on night baseline v93n: "add a little bit of yellow to the light sources like
v50m [the old night look] but not too yellow, a very subtle shade". Measured Town05 f450 (Lab b, +=yellow):
v51 highlights +15 / lit mid-tones +12; v93n +6 / +4. This adds about half that gap.

Per pixel: weight w = ramp(L, LO..HI) ** 1.2, times (1 - chroma/CHROMA_MAX) so already-coloured light
(red tail lights, green traffic lights) is left alone; b += WARM_B * w, a += WARM_A * w.
  usage: warm_lights.py <in_video> <out_video>   |   --frame N <in_video> <out.png>
  env:   WARM_B=6 WARM_A=1 LO=50 HI=200 CHROMA_MAX=30
"""
import os, sys
import cv2, numpy as np

from carla2real.common.vidcodec import fourcc_for
WB, WA = float(os.environ.get('WARM_B', '6')), float(os.environ.get('WARM_A', '1'))
LO, HI, CM = float(os.environ.get('LO', '50')), float(os.environ.get('HI', '200')), float(os.environ.get('CHROMA_MAX', '30'))


def process(im):
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
    w = np.clip((lab[..., 0] - LO) / (HI - LO), 0, 1) ** 1.2
    chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    w *= np.clip(1 - chroma / CM, 0, 1)
    w = cv2.GaussianBlur(w, (0, 0), 1.5)
    lab[..., 2] += WB * w
    lab[..., 1] += WA * w
    return cv2.cvtColor(np.clip(np.round(lab), 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[0] == '--frame':
        c = cv2.VideoCapture(a[2]); c.set(cv2.CAP_PROP_POS_FRAMES, int(a[1]))
        cv2.imwrite(a[3], process(c.read()[1])); sys.exit(0)
    cap = cv2.VideoCapture(a[0]); W, H = int(cap.get(3)), int(cap.get(4))
    vw = cv2.VideoWriter(a[1], fourcc_for(a[1]), cap.get(5) or 30, (W, H)); n = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        vw.write(process(f)); n += 1
    vw.release(); print(f'wrote {a[1]}: {n} frames (warm b {WB}, a {WA})')

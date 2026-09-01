#!/usr/bin/env python3
"""Compare de-shimmer variants PER CLASS, because the whole point is where the effect lands.

A single whole-frame flicker number cannot distinguish "smoothed the right things" from "smoothed
everything". The global filter's failure mode is precisely that it blurs the road (38% of the
frame, mean flicker 8.6 -- already stable) in order to reach the buildings (25% of the frame, mean
flicker 48.0 -- the actual source). So report both classes separately:

  road      sharpness should NOT drop. This is the largest surface, it barely flickers, and the
            perception stack keys on its lane markings.
  building  flicker should drop. This is where the visible shimmer lives.

  usage: deshim_compare.py <label_dir> <name>=<video> [<name>=<video> ...]
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image

LABDIR = sys.argv[1]
ARGS = sys.argv[2:]
ROAD, BUILDING = 13, 17
START, N = 300, 45

labs = sorted([f for f in os.listdir(LABDIR) if f.endswith('.png')])


def load(path):
    c = cv2.VideoCapture(path)
    c.set(cv2.CAP_PROP_POS_FRAMES, START)
    fr = []
    for _ in range(N):
        ok, f = c.read()
        if not ok:
            break
        fr.append(cv2.resize(f, (960, 480)))
    c.release()
    return fr


masks = []
for i in range(START, START + N):
    if i >= len(labs):
        break
    lab = np.array(Image.open(os.path.join(LABDIR, labs[i])))
    lab = lab if lab.ndim == 2 else lab[:, :, 0]
    masks.append(cv2.resize(lab, (960, 480), interpolation=cv2.INTER_NEAREST))

print(f'{"variant":<26}{"road sharp":>12}{"road flick":>12}{"bldg sharp":>12}{"bldg flick":>12}')
base = None
for a in ARGS:
    name, path = a.split('=', 1)
    fr = load(path)
    if len(fr) < 20:
        print(f'{name:<26}  -- missing --'); continue
    g = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in fr]
    st = np.stack(g)
    alt = np.abs(2 * st[1:-1] - st[:-2] - st[2:])
    mm = np.stack(masks[:len(fr)])[1:-1]
    out = []
    for cid in (ROAD, BUILDING):
        m = mm == cid
        # sharpness inside the class only, so one class's detail cannot mask the other's mush
        sh = []
        for k, x in enumerate(g[1:-1]):
            mk = m[k]
            if mk.sum() < 5000:
                continue
            lap = cv2.Laplacian(x.astype(np.uint8), cv2.CV_64F)
            sh.append(float(lap[mk].var()))
        out.append(float(np.mean(sh)) if sh else float('nan'))
        out.append(float(np.percentile(alt[m], 99)) if m.any() else float('nan'))
    print(f'{name:<26}{out[0]:12.1f}{out[1]:12.1f}{out[2]:12.1f}{out[3]:12.1f}')
print('\nwant: road sharpness held, building flicker down.')

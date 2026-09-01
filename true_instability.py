#!/usr/bin/env python3
"""Separate REAL temporal instability from detail-in-motion.

Every metric used so far measures |2x(t) - x(t-1) - x(t+1)| directly, and that conflates two very
different things: a surface whose texture is genuinely being reinvented each frame, and a
stationary detailed surface sweeping past a moving camera. The second is not a defect -- it is
what detail looks like in motion.

The proof that this matters: measured on building pixels, the CARLA source scores 56.1 and the
render scores 45.6. CARLA is a deterministic renderer whose facades are perfectly stable, so a
PERFECT render would score WORSE than the current one. Every filter that improved this number did
so by destroying detail, which is exactly the reported defect ("lost all the details").

The fix is to compare frame t against frame t-1 WARPED onto it by optical flow. Detail that simply
moved lands on itself and cancels; texture that was reinvented does not. What remains is the
instability that a viewer perceives as shimmer.

Reported per source: warped residual (the real defect), plain alternation (the old, misleading
number) and detail, so the difference between the two metrics is visible.

  usage: true_instability.py <label_ids_csv> <name>=<video|renderdir> ...
"""
from config import DATA
import glob
import os
import sys

import cv2
import numpy as np
from PIL import Image

LBL = os.path.join(DATA, 'training_v12_mapillary/test_Town10HD_sunny_inst_gt_label')
IDS = [int(x) for x in sys.argv[1].split(',')]
START, N = 300, 40
lbs = sorted(glob.glob(f'{LBL}/*.png'))


def frames(spec):
    if os.path.isdir(spec):
        fs = sorted(glob.glob(f'{spec}/*_synthesized_image.jpg')) or sorted(glob.glob(f'{spec}/*.png'))
        out = []
        for f in fs[START:START + N]:
            im = cv2.imread(f)
            if 'recorded' in spec:                      # CARLA writes BGR-as-RGB
                im = np.ascontiguousarray(im[:, :, ::-1])
            out.append(cv2.resize(im, (960, 480)))
        return out
    c = cv2.VideoCapture(spec)
    c.set(cv2.CAP_PROP_POS_FRAMES, START)
    out = []
    for _ in range(N):
        ok, f = c.read()
        if not ok:
            break
        out.append(cv2.resize(f, (960, 480)))
    c.release()
    return out


masks = []
for i in range(START, START + N):
    lab = np.array(Image.open(lbs[i]))
    lab = lab if lab.ndim == 2 else lab[:, :, 0]
    lab = cv2.resize(lab, (960, 480), interpolation=cv2.INTER_NEAREST)
    masks.append(cv2.erode(np.isin(lab, IDS).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)

print(f'{"source":<26}{"warped resid":>14}{"plain alt":>11}{"detail":>10}')
for a in sys.argv[2:]:
    name, spec = a.split('=', 1)
    fr = frames(spec)
    if len(fr) < 10:
        print(f'{name:<26}  -- missing --'); continue
    g = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in fr]
    resid, det = [], []
    for k in range(1, len(g)):
        m = masks[k]
        if m.sum() < 5000:
            continue
        fl = cv2.calcOpticalFlowFarneback(g[k - 1].astype(np.uint8), g[k].astype(np.uint8),
                                          None, 0.5, 3, 21, 3, 5, 1.2, 0)
        h, w = g[k].shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        # warp t-1 onto t: detail that only moved now sits on itself and cancels
        warped = cv2.remap(g[k - 1], xx + fl[..., 0], yy + fl[..., 1],
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        # ignore where the warp is untrustworthy (occlusion, frame edge)
        ok_ = (np.abs(warped - g[k]) < 120) & m
        if ok_.sum() > 5000:
            resid.append(float(np.abs(warped - g[k])[ok_].mean()))
        det.append(float(cv2.Laplacian(g[k].astype(np.uint8), cv2.CV_64F)[m].var()))
    st = np.stack(g)
    alt = np.abs(2 * st[1:-1] - st[:-2] - st[2:])
    mm = np.stack(masks[1:len(g) - 1])
    print(f'{name:<26}{np.mean(resid):14.2f}{float(alt[mm].mean()):11.2f}{np.mean(det):10.1f}')
print('\nwarped residual is the real defect; plain alt rewards blur and penalises detail.')

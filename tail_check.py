#!/usr/bin/env python3
"""Check a model at the TAIL of a full-length render, against the v50 parent at the same frames.

Both reported defects are invisible early:
  - white dots accumulate through the autoregressive loop (clean at frame 200, ruined by 900)
  - sky ghosts are subtle and easiest to see against bright uniform regions late in the drive
So this samples frames 880-920 only, and always compares to the parent at the SAME frame indices,
because road texture varies enormously with scene content (a busy frame reads 4x a quiet one for
every model, which previously looked like divergence and was not).

  sky ghost = gradient energy inside SKY-labelled pixels. Sky should be smooth; any structure
              there is the temporal loss having painted warped building edges into it.
  specks    = isolated pixels far brighter than their local neighbourhood (the white dots).
  road tex  = high-pass energy on ROAD-labelled pixels (the crosshatch weave).

  usage: tail_check.py <model_ckpt> <phase>
"""
import os
from config import DATA, RESULTS
import cv2, glob, os, sys
import numpy as np

R = RESULTS
LBLROOT = os.path.join(DATA, 'training_v12_mapillary')
M, PHASE = sys.argv[1], sys.argv[2]
PARENT = 'carla2real_semantic_v50_graft'
ROAD, SKY = 13, 27
LO, N = 880, 20


def measure(model, phase):
    lbls = sorted(glob.glob(f'{LBLROOT}/{phase}_label/*.png'))
    fs = sorted(glob.glob(f'{R}/{model}/{phase}_latest/images/*_synthesized_image.jpg'))[LO:LO + N]
    if len(fs) < 5:
        return None
    road, sky, sp = [], [], []
    for f in fs:
        idx = int(os.path.basename(f).split('_')[0])
        g = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)
        gf = g.astype(np.float32)
        gi = g.astype(np.int16)
        sp.append(int(((gi - cv2.medianBlur(g, 5).astype(np.int16)) > 40).sum()))
        if idx < len(lbls):
            lab = cv2.resize(cv2.imread(lbls[idx], cv2.IMREAD_GRAYSCALE), (g.shape[1], g.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
            hp = gf - cv2.GaussianBlur(gf, (0, 0), 2)
            k = np.ones((9, 9), np.uint8)
            rm = cv2.erode((lab == ROAD).astype(np.uint8), k) > 0
            sm = cv2.erode((lab == SKY).astype(np.uint8), k) > 0
            if rm.sum() > 20000:
                road.append(float((hp[rm] ** 2).mean()))
            if sm.sum() > 20000:
                sky.append(float((hp[sm] ** 2).mean()))
    f2 = lambda v: float(np.mean(v)) if v else float('nan')
    return f2(road), f2(sky), float(np.mean(sp))


a = measure(M, PHASE)
b = measure(PARENT, PHASE)
if not a or not b:
    print('  tail_check: missing renders'); raise SystemExit(1)
print(f'  {"":<22}{"road tex":>10}{"sky ghost":>11}{"specks":>10}')
print(f'  {"v50 parent":<22}{b[0]:10.2f}{b[1]:11.3f}{b[2]:10.0f}')
print(f'  {M.replace("carla2real_semantic_",""):<22}{a[0]:10.2f}{a[1]:11.3f}{a[2]:10.0f}')
print(f'  {"ratio vs parent":<22}{a[0]/b[0]:9.2f}x{a[1]/b[1]:10.2f}x{a[2]/b[2]:9.2f}x')
ok = (a[0] / b[0] <= 1.2) and (a[1] / b[1] <= 1.2) and (a[2] / b[2] <= 1.2)
print(f'\n  VERDICT: {"PASS - no worse than the parent on any of the three" if ok else "FAIL - worse than the parent on at least one"}')

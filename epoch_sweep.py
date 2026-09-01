#!/usr/bin/env python3
"""Score any render directory on both axes at once: the vegetation gain and the three side costs.

v63's final epoch bought the vegetation fix (far detail +19%, far/near 0.83 -> 1.00) but pushed
road texture to 1.81x the parent, sky ghost 1.57x and specks 1.24x -- past the 1.2x gate. The
weighting raised high-frequency output everywhere, not only on vegetation.

Checkpoints exist every 2 epochs, so the trade-off can be read along the trajectory that was
already trained rather than by guessing new weights and running again. This measures an arbitrary
results directory with exactly the definitions veg_report.py and tail_check.py use, so the numbers
line up with what is already recorded.

  usage: epoch_sweep.py <results_dir> [<results_dir> ...]
"""
from config import DATA, RESULTS
import glob
import os
import sys

import cv2
import numpy as np
from PIL import Image

LBLROOT = os.path.join(DATA, 'training_v12_mapillary')
PHASE = 'test_Town10HD_sunny_inst_gt'
LBL = f'{LBLROOT}/{PHASE}_label'
VEG, ROAD, SKY = [30], 13, 27
V_START, V_N = 300, 30          # veg_report.py window
T_LO, T_N = 880, 20             # tail_check.py window
lbs = sorted(glob.glob(f'{LBL}/*.png'))


def load_lab(i, shape):
    lab = np.array(Image.open(lbs[i]))
    lab = lab if lab.ndim == 2 else lab[:, :, 0]
    return cv2.resize(lab, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def veg(fs):
    near, far = [], []
    for i in range(V_START, V_START + V_N):
        im = cv2.resize(cv2.imread(fs[i]), (1920, 960))
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        lab = load_lab(i, (960, 1920))
        nl, cc, st, _ = cv2.connectedComponentsWithStats(np.isin(lab, VEG).astype(np.uint8), 8)
        lap = cv2.Laplacian(g, cv2.CV_64F)
        for c in range(1, nl):
            a = st[c, cv2.CC_STAT_AREA]
            if a < 150:
                continue
            v = float(lap[cc == c].var())
            (near if a >= 4000 else far).append(v)
    return float(np.mean(near)), float(np.mean(far))


def tail(fs):
    road, sky, sp = [], [], []
    for f in fs[T_LO:T_LO + T_N]:
        idx = int(os.path.basename(f).split('_')[0])
        g = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)
        gf, gi = g.astype(np.float32), g.astype(np.int16)
        sp.append(int(((gi - cv2.medianBlur(g, 5).astype(np.int16)) > 40).sum()))
        lab = load_lab(idx, g.shape)
        hp = gf - cv2.GaussianBlur(gf, (0, 0), 2)
        k = np.ones((9, 9), np.uint8)
        rm = cv2.erode((lab == ROAD).astype(np.uint8), k) > 0
        sm = cv2.erode((lab == SKY).astype(np.uint8), k) > 0
        if rm.sum() > 20000:
            road.append(float((hp[rm] ** 2).mean()))
        if sm.sum() > 20000:
            sky.append(float((hp[sm] ** 2).mean()))
    return float(np.mean(road)), float(np.mean(sky)), float(np.mean(sp))


R = RESULTS
PARENT = f'{R}/carla2real_semantic_v50_graft/{PHASE}_latest/images'
rows = []
for d in [PARENT] + sys.argv[1:]:
    fs = sorted(glob.glob(f'{d.rstrip("/")}/*_synthesized_image.jpg'))
    if len(fs) < T_LO + T_N:
        print(f'{d}: only {len(fs)} frames -- skipped')
        continue
    n, f_ = veg(fs)
    r, s, k = tail(fs)
    rows.append((d, n, f_, r, s, k))

base = rows[0]
print(f'{"render":<14}{"veg near":>9}{"veg far":>9}{"far/near":>9}'
      f'{"road":>8}{"sky":>7}{"specks":>8}   gate")')
for d, n, f_, r, s, k in rows:
    tag = 'parent' if d == PARENT else os.path.basename(os.path.dirname(d)).replace(PHASE + '_', 'ep')
    rr, ss, kk = r / base[3], s / base[4], k / base[5]
    gate = 'PASS' if max(rr, ss, kk) <= 1.2 else 'FAIL'
    print(f'{tag:<14}{n:9.0f}{f_:9.0f}{f_/n:9.2f}'
          f'{rr:7.2f}x{ss:6.2f}x{kk:7.2f}x   {gate}')
print('\nwanted: far up and far/near toward 1.0, with all three ratios inside 1.2x.')

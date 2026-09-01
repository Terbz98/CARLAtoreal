#!/usr/bin/env python3
"""Where are v63's extra specks?

tail_check counts isolated pixels >40 above their local median over the WHOLE frame, because that
is how the white-dot defect first showed up. But a sharp leaf highlight passes the same test, so a
model that genuinely sharpens foliage will raise the count without having the defect. Split the
count by label class: if the extra specks sit on vegetation they are the intended detail; if they
sit on road, sky or car bodies -- smooth surfaces with nothing to highlight -- they are the defect.

  usage: speck_where.py <results_dir_a> <results_dir_b>
"""
from config import DATA
import glob
import os
import sys

import cv2
import numpy as np
from PIL import Image

LBL = os.path.join(DATA, 'training_v12_mapillary/test_Town10HD_sunny_inst_gt_label')
lbs = sorted(glob.glob(f'{LBL}/*.png'))
GROUPS = {'road': [13, 24], 'sky': [27], 'vegetation': [30], 'building': [17, 6], 'vehicle': [55, 61]}
LO, N = 880, 20


def count(d):
    fs = sorted(glob.glob(f'{d.rstrip("/")}/*_synthesized_image.jpg'))[LO:LO + N]
    tot = {k: 0 for k in GROUPS}
    area = {k: 0 for k in GROUPS}
    for f in fs:
        i = int(os.path.basename(f).split('_')[0])
        g = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)
        hit = (g.astype(np.int16) - cv2.medianBlur(g, 5).astype(np.int16)) > 40
        lab = np.array(Image.open(lbs[i]))
        lab = lab if lab.ndim == 2 else lab[:, :, 0]
        lab = cv2.resize(lab, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
        for k, ids in GROUPS.items():
            m = np.isin(lab, ids)
            tot[k] += int(hit[m].sum())
            area[k] += int(m.sum())
    return tot, area


a, area = count(sys.argv[1])
b, _ = count(sys.argv[2])
print(f'{"class":<12}{"parent":>10}{"v63":>10}{"delta":>10}{"per Mpx parent":>16}{"per Mpx v63":>13}')
for k in GROUPS:
    pa = 1e6 * a[k] / max(area[k], 1)
    pb = 1e6 * b[k] / max(area[k], 1)
    print(f'{k:<12}{a[k]:10d}{b[k]:10d}{b[k]-a[k]:+10d}{pa:16.0f}{pb:13.0f}')
print('\nextra specks on vegetation = the sharpening that was asked for;')
print('extra specks on road, sky or vehicles = the white-dot defect.')

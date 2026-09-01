#!/usr/bin/env python3
"""Score a model on the vegetation deficit, split by distance, against the real-photo ceiling.

The two numbers that define the problem, measured on vegetation-labelled pixels:
    real training photos   near 1191   far 1246   far/near 1.05
    CARLA source            near  928   far 1030   far/near 1.11
    v50 render              near  672   far  561   far/near 0.83
So there are two things to fix and both must be reported: the ABSOLUTE detail (the render reaches
56% of what the photographs carry) and the far/near RATIO (the render is the only source that
loses detail with distance).

Deliberately not a single scalar. A model could raise near detail and leave distant trees exactly
as bad, and one averaged number would call that a success.

Distance is proxied by connected-region area -- a distant tree occupies few pixels -- because the
depth map is a monocular estimate and not reliable enough to bucket on.

  usage: veg_report.py <model_ckpt> [<model_ckpt> ...]
"""
import os
from config import DATA, RESULTS
import glob
import sys

import cv2
import numpy as np
from PIL import Image

R = RESULTS
LBL = os.path.join(DATA, 'training_v12_mapillary/test_Town10HD_sunny_inst_gt_label')
PHS = 'test_Town10HD_sunny_inst_gt_latest'
VEG = [30]
START, N = 300, 30
lbs = sorted(glob.glob(f'{LBL}/*.png'))

CEILING = ('REAL training photos', 1191.5, 1245.9)
REFS = [('CARLA source', 927.9, 1029.9), ('v50 parent', 672.5, 561.0)]


def split_detail(gray, mask):
    near, far = [], []
    nl, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    for i in range(1, nl):
        a = st[i, cv2.CC_STAT_AREA]
        if a < 150:
            continue
        v = float(lap[lab == i].var())
        (near if a >= 4000 else far).append(v)
    return near, far


print(f'{"source":<26}{"near":>9}{"far":>9}{"far/near":>10}{"% of photo ceiling":>20}')
print(f'{CEILING[0]:<26}{CEILING[1]:9.1f}{CEILING[2]:9.1f}{CEILING[2]/CEILING[1]:10.2f}{"100%":>20}')
for name, n, f in REFS:
    print(f'{name:<26}{n:9.1f}{f:9.1f}{f/n:10.2f}{100*(n+f)/(CEILING[1]+CEILING[2]):19.0f}%')

for model in sys.argv[1:]:
    fs = sorted(glob.glob(f'{R}/{model}/{PHS}/images/*_synthesized_image.jpg'))
    if len(fs) < START + N:
        print(f'{model[:25]:<26}   -- not rendered --')
        continue
    near, far = [], []
    for i in range(START, START + N):
        im = cv2.resize(cv2.imread(fs[i]), (1920, 960))
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        lab = np.array(Image.open(lbs[i]))
        lab = lab if lab.ndim == 2 else lab[:, :, 0]
        lab = cv2.resize(lab, (1920, 960), interpolation=cv2.INTER_NEAREST)
        a, b = split_detail(g, np.isin(lab, VEG))
        near += a; far += b
    n = float(np.mean(near)) if near else float('nan')
    f = float(np.mean(far)) if far else float('nan')
    label = model.replace('carla2real_semantic_', '')
    print(f'{label:<26}{n:9.1f}{f:9.1f}{f/n:10.2f}{100*(n+f)/(CEILING[1]+CEILING[2]):19.0f}%')
print('\nboth columns matter: raising near detail while distant trees stay bad is not a fix.')

#!/usr/bin/env python3
"""Compare temporal-model variants on speckle, TRUE sharpness, and flicker.

Why this exists: plain Laplacian variance was measuring the defect. Across v54's epochs it went
210 -> 265 -> 286 -> 327 while isolated bright pixels went 2711 -> 5377 -> 10205 -> 12212. The
"+80% sharpness" reported for v54 was substantially white speckle, not detail.

So sharpness is measured TWICE:
  raw    Laplacian variance of the frame            -- inflated by speckle
  clean  Laplacian variance after a 3x3 median      -- a median removes isolated single-pixel
         outliers while leaving edges and texture intact, so this is structure only
The gap between them is itself a speckle indicator, and `clean` is the number to judge detail on.

specks = pixels more than 40 grey levels brighter than their 5x5 local median: an isolated bright
dot, which is what shows on screen. Real highlights are not isolated and survive the median.
"""
from config import RESULTS
import cv2, glob
import numpy as np

R = RESULTS
SUN = 'test_Town10HD_sunny_inst_gt'
NIG = 'test_Town03_night_inst_gt'
CASES = [
    ('v50  parent', 'carla2real_semantic_v50_graft', SUN, 'latest'),
    ('v54  temporal ep1', 'carla2real_semantic_v54_tsunny', SUN, '1'),
    ('v54  temporal ep3', 'carla2real_semantic_v54_tsunny', SUN, 'latest'),
    ('v57a no video-disc', 'carla2real_semantic_v57a_novgan', SUN, 'latest'),
    ('v57b vgan 0.1', 'carla2real_semantic_v57b_lowvgan', SUN, 'latest'),
    ('v51  parent (night)', 'carla2real_semantic_v51_night', NIG, 'latest'),
    ('v56  temporal (night)', 'carla2real_semantic_v56_tnight2', NIG, 'latest'),
]
N = 60

print(f'{"model":<24}{"specks":>9}{"raw sh":>9}{"clean sh":>10}{"alt p99":>9}')
for name, m, phase, ep in CASES:
    d = f'{R}/{m}/{phase}_{ep}/images'
    fs = sorted(glob.glob(f'{d}/*_synthesized_image.jpg'))[:N]
    if len(fs) < 10:
        print(f'{name:<24}{"-- not rendered --":>28}')
        continue
    sp, raw, cln, grey = [], [], [], []
    for f in fs:
        g = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)
        gi = g.astype(np.int16)
        med5 = cv2.medianBlur(g, 5).astype(np.int16)
        sp.append(int(((gi - med5) > 40).sum()))
        raw.append(cv2.Laplacian(g, cv2.CV_64F).var())
        cln.append(cv2.Laplacian(cv2.medianBlur(g, 3), cv2.CV_64F).var())
        grey.append(cv2.resize(g, (480, 240)).astype(np.float32))
    a = np.stack(grey)
    alt = np.abs(2 * a[1:-1] - a[:-2] - a[2:])
    print(f'{name:<24}{np.mean(sp):9.0f}{np.mean(raw):9.1f}{np.mean(cln):10.1f}'
          f'{float(np.percentile(alt.mean(axis=0), 99)):9.2f}')
print('\nclean sh is the honest detail number; raw sh counts the dots as if they were texture.')

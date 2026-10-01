#!/usr/bin/env python3
"""Flatten one PandaSet camera into a flat, index-ordered image directory, daylight frames only.

WHY A SECOND FLATTENER. flatten_pandaset.py hard-codes 'front_camera' and takes every frame. Both
choices turn out to cost real quality:

  * PandaSet ships SIX cameras and only the front one was ever extracted, so the best-licensed
    source in the corpus has been capped at 8,240 frames for no reason other than the glob. The two
    forward-oblique cameras are the same rig, the same drives and the same street scenes. Measured
    at a common 1024 width, daylight frames only:

        camera                detail    sat
        front_camera            6.27   39.1
        front_left_camera       5.39   48.7
        front_right_camera      6.61   48.5

    front_right is the sharpest source in the project and both carry ~9 more saturation than the
    front camera. left_camera, right_camera and back_camera are rejected: they look across the
    street rather than along it, and the back camera is a third ego bonnet.

  * 17.5% of PandaSet is shot at night and 1.9% at dusk, and build_pandaset_corpus.sh puts all of
    it into the SUNNY corpus. The night model has its own corpus, so those frames are contamination
    in this one and they pull exactly the way the sunny complaint points: dark and grey.

THE DAYLIGHT GATE IS ON THE SKY, NOT THE GROUND. A ground-contrast score is satisfied by any dark
scene, which is the bug select_sunlit.py already had to fix once -- its first version "did not find
sunlit frames, it found the dimmest frames in the dataset" and cost v84. The same trap caught the
survey that led here: scored on ground contrast, front_right reads 43% hard sun, and the frames it
picks are after dark or have a black car filling the near field. The sky band is the honest test of
whether it is daytime, so that is what this gates on.

  usage: flatten_pandaset_cam.py <camera> <out_dir> <prefix> [--sky 90] [--no-gate]
"""
import os
import sys

import cv2
import numpy as np

from carla2real.config import DATA
PS = os.path.join(DATA, 'pandaset', 'extracted', 'pandaset')
CAM, OUT, PREFIX = sys.argv[1], sys.argv[2], sys.argv[3]
SKY = float(sys.argv[sys.argv.index('--sky') + 1]) if '--sky' in sys.argv else 90.0
GATE = '--no-gate' not in sys.argv

os.makedirs(OUT, exist_ok=True)

files = []
for scene in sorted(os.listdir(PS)):
    d = os.path.join(PS, scene, 'camera', CAM)
    if not os.path.isdir(d):
        continue
    names = [f for f in os.listdir(d) if f.lower().endswith('.jpg')]
    # PandaSet frames are 00.jpg..79.jpg, so a lexical sort puts 10 before 2 and scrambles the
    # sequences the temporal corpus builder depends on
    names.sort(key=lambda x: int(os.path.splitext(x)[0]))
    files += [os.path.join(d, f) for f in names]

if not files:
    raise SystemExit(f'no images under {PS}/*/camera/{CAM} -- is that camera extracted?')

n = dark = 0
for p in files:
    im = cv2.imread(p)
    if im is None:
        continue
    if GATE:
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        if float(np.percentile(g[:int(g.shape[0] * 0.25)], 75)) < SKY:
            dark += 1
            continue
    im = cv2.resize(im, (1024, 512), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(OUT, f'{PREFIX}{n:05d}.jpg'), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
    n += 1

print(f'  {CAM}: wrote {n} daylight frames as {PREFIX}*, dropped {dark} dark ones '
      f'({100.0 * dark / max(1, n + dark):.1f}%)')
if n < 4000:
    raise SystemExit(f'expected well over 4000 daylight frames, got {n} -- refusing to report success')

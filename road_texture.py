#!/usr/bin/env python3
"""Measure texture energy on pixels the LABEL MAP says are road.

Motivation: the crosshatch that ruined v54 is a weave painted over surfaces that should be
smooth. Every generic image metric failed on it --

  alt p99 (flicker)   said v54 was CALMER   -- a static grid is perfectly frame-to-frame
                                               consistent, so a flicker metric REWARDS it
  Laplacian sharpness said v54 was SHARPER  -- a grid is high-frequency, so it reads as detail
  median-cleaned      said v54 was SHARPER  -- a median kills isolated specks, not a structure
  FFT peak ratio      barely moved (79.9 -> 87.3) for a night-and-day visual difference

-- because all of them ask "how much high-frequency content is there", and the artifact IS
high-frequency content. The question that actually separates them is "how much high-frequency
content is somewhere it does not belong", and the label map answers that for free: road is a
large, genuinely smooth surface, so any texture energy there is either real asphalt grain (small
and consistent across models) or artifact.

Comparing against the PARENT on the same labels is what makes it interpretable: both models
render identical inputs, so a ratio well above 1 means the fine-tune added texture to the road.

  usage: road_texture.py <phase> <model_ckpt> [<model_ckpt> ...]
"""
import os
from config import DATA, RESULTS
import cv2, glob, os, sys
import numpy as np

R = RESULTS
LBL = os.path.join(DATA, 'training_v12_mapillary')
ROAD = 13          # Mapillary-65 road
N = 20

phase = sys.argv[1]
models = sys.argv[2:]
lbls = sorted(glob.glob(f'{LBL}/{phase.rsplit("_", 1)[0]}_label/*.png')) if False else \
       sorted(glob.glob(f'{LBL}/{"_".join(phase.split("_")[:-1])}_label/*.png'))

print(f'phase {phase}   road-labelled pixels only, {N} frames')
print(f'{"model":<38}{"road texture":>13}{"vs parent":>11}')
base = None
for m in models:
    fs = sorted(glob.glob(f'{R}/{m}/{phase}/images/*_synthesized_image.jpg'))
    if len(fs) < N + 200 or not lbls:
        print(f'{m.replace("carla2real_semantic_",""):<38}{"-- missing --":>13}')
        continue
    vals = []
    for f in fs[200:200 + N]:
        idx = int(os.path.basename(f).split('_')[0])
        if idx >= len(lbls):
            continue
        g = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY).astype(np.float32)
        lab = cv2.imread(lbls[idx], cv2.IMREAD_GRAYSCALE)
        if lab is None:
            continue
        lab = cv2.resize(lab, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
        # erode so we never include the boundary between road and kerb/car, where a real edge
        # would masquerade as texture
        mask = cv2.erode((lab == ROAD).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        if mask.sum() < 20000:
            continue
        hp = g - cv2.GaussianBlur(g, (0, 0), 2)      # high-pass: texture, not shading
        vals.append(float((hp[mask] ** 2).mean()))
    if not vals:
        print(f'{m.replace("carla2real_semantic_",""):<38}{"-- no road --":>13}')
        continue
    v = float(np.mean(vals))
    if base is None:
        base = v
    print(f'{m.replace("carla2real_semantic_",""):<38}{v:13.1f}{v / base:10.2f}x')
print('\nratio ~1 = same surface smoothness as the parent; >1.5 = the model added texture to road.')

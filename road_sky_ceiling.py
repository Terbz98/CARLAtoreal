#!/usr/bin/env python3
"""Is v63's extra road grain a defect, or movement toward what a real photograph looks like?

tail_check compares to the v50 parent, which is the right gate for catching a NEW artefact but
says nothing about which side of the reference the parent sits on. The vegetation work already
showed the parent is far below the photographs it was trained on. So measure road high-pass energy
the same way tail_check does, but on the real training photos and the CARLA source too.

Sky is handled separately: the sky mask is eroded by 9x9 in tail_check, which is not enough to
exclude wires, palm fronds and banner edges -- thin structures the label map does not resolve, and
which v63 renders far more crisply. Erode by 31x31 as well: if the gap closes, the sky ghost signal
is structures spilling into sky-labelled pixels, not ghosting.
"""
from config import DATA, RESULTS
import glob
import os

import cv2
import numpy as np
from PIL import Image

D = DATA
R = RESULTS
PHASE = 'test_Town10HD_sunny_inst_gt'
LBL = f'{D}/training_v12_mapillary/{PHASE}_label'
ROAD, SKY = 13, 27
lbs = sorted(glob.glob(f'{LBL}/*.png'))
carla = sorted(glob.glob(f'{D}/recorded_Town10HD_sunny_inst/rgb/*.png'))


def hp_energy(g, mask):
    gf = g.astype(np.float32)
    hp = gf - cv2.GaussianBlur(gf, (0, 0), 2)
    return float((hp[mask] ** 2).mean()) if mask.sum() > 20000 else None


def render_rows(d, idxs):
    fs = sorted(glob.glob(f'{d}/*_synthesized_image.jpg'))
    road, sky9, sky31 = [], [], []
    for i in idxs:
        g = cv2.cvtColor(cv2.imread(fs[i]), cv2.COLOR_BGR2GRAY)
        lab = cv2.resize(cv2.imread(lbs[i], cv2.IMREAD_GRAYSCALE), (g.shape[1], g.shape[0]),
                         interpolation=cv2.INTER_NEAREST)
        for acc, ids, k in ((road, ROAD, 9), (sky9, SKY, 9), (sky31, SKY, 31)):
            m = cv2.erode((lab == ids).astype(np.uint8), np.ones((k, k), np.uint8)) > 0
            v = hp_energy(g, m)
            if v is not None:
                acc.append(v)
    f = lambda a: float(np.mean(a)) if a else float('nan')
    return f(road), f(sky9), f(sky31)


def carla_rows(idxs):
    road, sky9, sky31 = [], [], []
    for i in idxs:
        im = np.ascontiguousarray(cv2.imread(carla[i])[:, :, ::-1])   # CARLA writes BGRA
        g = cv2.cvtColor(cv2.resize(im, (2048, 1024)), cv2.COLOR_BGR2GRAY)
        lab = cv2.resize(cv2.imread(lbs[i], cv2.IMREAD_GRAYSCALE), (2048, 1024),
                         interpolation=cv2.INTER_NEAREST)
        for acc, ids, k in ((road, ROAD, 9), (sky9, SKY, 9), (sky31, SKY, 31)):
            m = cv2.erode((lab == ids).astype(np.uint8), np.ones((k, k), np.uint8)) > 0
            v = hp_energy(g, m)
            if v is not None:
                acc.append(v)
    f = lambda a: float(np.mean(a)) if a else float('nan')
    return f(road), f(sky9), f(sky31)


def photo_rows(n=120):
    imgs = sorted(glob.glob(f'{D}/training_v49_chroma/train_img/mvhr_*.jpg'))[:n]
    road, sky9, sky31 = [], [], []
    for p in imgs:
        stem = os.path.basename(p).rsplit('.', 1)[0]
        lp = f'{D}/training_v49_chroma/train_label/{stem}.png'
        if not os.path.exists(lp):
            continue
        im = cv2.imread(p)
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        lab = np.array(Image.open(lp))
        lab = lab if lab.ndim == 2 else lab[:, :, 0]
        if lab.shape != g.shape:
            lab = cv2.resize(lab, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
        for acc, ids, k in ((road, ROAD, 9), (sky9, SKY, 9), (sky31, SKY, 31)):
            m = cv2.erode((lab == ids).astype(np.uint8), np.ones((k, k), np.uint8)) > 0
            v = hp_energy(g, m)
            if v is not None:
                acc.append(v)
    f = lambda a: float(np.mean(a)) if a else float('nan')
    return f(road), f(sky9), f(sky31)


idxs = range(880, 900)
rows = [
    ('REAL training photos', photo_rows()),
    ('CARLA source', carla_rows(idxs)),
    ('v50 parent', render_rows(f'{R}/carla2real_semantic_v50_graft/{PHASE}_latest/images', idxs)),
    ('v63_veg', render_rows(f'{R}/carla2real_semantic_v63_veg/{PHASE}_latest/images', idxs)),
]
print(f'{"source":<24}{"road hp":>10}{"sky hp e9":>11}{"sky hp e31":>12}')
for name, (r, s9, s31) in rows:
    print(f'{name:<24}{r:10.2f}{s9:11.3f}{s31:12.3f}')
print('\nroad: if the parent sits far below the photographs, more grain is movement toward them.')
print('sky e31 vs e9: if v63 closes the gap under the harder erode, the ghost was edge spill.')

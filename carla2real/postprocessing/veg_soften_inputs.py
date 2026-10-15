#!/usr/bin/env python3
"""Soften the CARLA geometry the generator is conditioned on inside vegetation (inference only).

WHY. The user: v90dv's trees look cartoonish. Diagnosed 2026-10-10 on Town05 f200: the horizontal
"brush streaks" on the pines come from CARLA's depth and normal maps, which carry the low-poly pine's
stacked branch layers. Zeroing edges alone left them; blurring depth+normals inside vegetation removes
them (sigma 12 over-smooths into dark blobs; sigma 3-4 keeps a natural conifer texture).

Per frame: vegetation mask (label 30, dilated 5 px, feathered 2 px); depth and normal blurred by SIGMA
there; edge channel zeroed there. Everything else is a symlink to the original.
  usage: veg_soften_inputs.py <src_root> <dst_root> <phase> [<phase> ...]    env: SIGMA=3.5
"""
import os, sys
import cv2, numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
SIG = float(os.environ.get('SIGMA', '3.5'))
SRC, DST, PHASES = sys.argv[1], sys.argv[2], sys.argv[3:]


def one(args):
    ph, f = args
    lab = np.array(Image.open(f'{SRC}/{ph}_label/{f}')); lab = lab if lab.ndim == 2 else lab[..., 0]
    veg = cv2.GaussianBlur(cv2.dilate((lab == 30).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32), (0, 0), 2)
    for c in ('depth', 'normal', 'edge'):
        p = f'{SRC}/{ph}_{c}/{f}'
        if not os.path.exists(p): continue
        raw = cv2.imread(p, cv2.IMREAD_UNCHANGED); im = raw.astype(np.float32)
        vm = cv2.resize(veg, (im.shape[1], im.shape[0]))
        if im.ndim == 3: vm = vm[..., None]
        out = im * (1 - vm) if c == 'edge' else im + (cv2.GaussianBlur(im, (0, 0), SIG) - im) * vm
        hi = np.iinfo(raw.dtype).max
        cv2.imwrite(f'{DST}/{ph}_{c}/{f}', np.clip(np.round(out), 0, hi).astype(raw.dtype))


os.makedirs(DST, exist_ok=True)
jobs = []
for ph in PHASES:
    for d in os.listdir(SRC):
        if not d.startswith(ph + '_'): continue
        c = d[len(ph) + 1:]
        if c in ('depth', 'normal', 'edge'):
            os.makedirs(f'{DST}/{d}', exist_ok=True)
        elif not os.path.lexists(f'{DST}/{d}'):
            os.symlink(os.path.realpath(f'{SRC}/{d}'), f'{DST}/{d}')
    jobs += [(ph, f) for f in sorted(os.listdir(f'{SRC}/{ph}_label'))]
with ProcessPoolExecutor(20) as ex: list(ex.map(one, jobs, chunksize=50))
print(f'softened {len(jobs)} frames (sigma {SIG})')

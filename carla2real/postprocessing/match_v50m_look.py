#!/usr/bin/env python3
"""Grade a sunny render toward v50m's look, class by class, keeping the render's own vehicles.

WHY. User, 2026-10-13: "make it as similar as possible to v50m like before but with the vehicle colour of
now". v50m is the look the user has preferred since September; the current chain's vehicle pass gives the
car colours they like. Same recording, same frame index, same labels, so v50m frame t is a pixel-aligned
reference for render frame t.

Per frame, for each static class (road, sidewalk, curb, building, wall, fence, pole, vegetation, terrain,
sky, lane marking): the class's Lab mean and standard deviation in the render are moved to v50m's
(affine per channel, std ratio clamped to 0.6..1.6), through a feathered class mask, by STRENGTH. Vehicles
and people are left exactly as rendered. Texture and structure stay the render's; only tone and colour move.
  usage: match_v50m_look.py <render.mp4> <v50m.mp4> <label_dir> <out.mp4>   env: STRENGTH=0.85
"""
import glob, os, sys
import cv2, numpy as np
from PIL import Image

from carla2real.common.vidcodec import fourcc_for

S = float(os.environ.get('STRENGTH', '0.85'))
CLASSES = [13, 15, 2, 17, 6, 3, 45, 30, 29, 27, 24, 23, 16]
KEEP = [52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 19, 20, 21, 22]   # vehicles, people


def grade(im, ref, lab):
    A = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
    R = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    out = A.copy()
    acc = np.zeros(A.shape[:2], np.float32)
    shift = np.zeros_like(A)
    for c in CLASSES:
        m = lab == c
        if m.sum() < 800:
            continue
        a, r = A[m], R[m]
        mu_a, sd_a = a.mean(0), a.std(0) + 1e-3
        mu_r, sd_r = r.mean(0), r.std(0) + 1e-3
        k = np.clip(sd_r / sd_a, 0.6, 1.6)
        w = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.5)
        target = (A - mu_a) * k + mu_r
        shift += w[..., None] * (target - A)
        acc += w
    keep = cv2.GaussianBlur(np.isin(lab, KEEP).astype(np.float32), (0, 0), 1.5)
    norm = np.maximum(acc, 1.0)[..., None]
    out = A + S * (1 - keep[..., None]) * shift / norm
    return cv2.cvtColor(np.clip(np.round(out), 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


if __name__ == '__main__':
    IN, REF, LABD, OUT = sys.argv[1:5]
    labs = sorted(glob.glob(os.path.join(LABD, '*.png')))
    a, b = cv2.VideoCapture(IN), cv2.VideoCapture(REF)
    W, H = int(a.get(3)), int(a.get(4))
    vw = cv2.VideoWriter(OUT, fourcc_for(OUT), a.get(5) or 30, (W, H)); n = 0
    while True:
        ok1, f = a.read(); ok2, g = b.read()
        if not (ok1 and ok2): break
        if n < len(labs):
            l = np.array(Image.open(labs[n])); l = l if l.ndim == 2 else l[..., 0]
            l = cv2.resize(l, (W, H), interpolation=cv2.INTER_NEAREST)
            f = grade(f, cv2.resize(g, (W, H)), l)
        vw.write(f); n += 1
    vw.release(); print(f'wrote {OUT}: {n} frames (strength {S})')

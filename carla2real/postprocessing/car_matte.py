#!/usr/bin/env python3
"""Take the plastic gloss off vehicles: compress body highlights toward what real cars show.

WHY. User, 2026-10-15: "the cars still have that glossy plastic look". Vehicle bodies (label car, eroded 3 px),
Lab L p98 minus p50: real ZOD photographs 80.6; v98f 152, v50m 147 -- highlights almost twice as far above the body
tone as on real cars, which reads as glossy plastic. Fine texture already matches (8.3 vs 8.6).

Per vehicle region (connected component): L above the region's median is scaled by K (soft knee), so specular
highlights come down while shading below the median, colour and edges are untouched. Lights (very bright, saturated
red/amber tail lights) are excluded by chroma.
  usage: car_matte.py <in.mp4> <label_dir> <out.mp4> | --frame N ...   env: K=0.55
"""
import glob, os, sys
import cv2, numpy as np
from PIL import Image

from carla2real.common.vidcodec import fourcc_for
K = float(os.environ.get('K', '0.55'))
VEH = [52, 54, 55, 56, 57, 58, 59, 60, 61, 62]


def process(im, lab):
    m = np.isin(lab, VEH).astype(np.uint8)
    if m.sum() < 500: return im
    A = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32); L = A[..., 0]
    chroma = np.hypot(A[..., 1] - 128, A[..., 2] - 128)
    n, cc, st, _ = cv2.connectedComponentsWithStats(m)
    newL = L.copy()
    for k in range(1, n):
        if st[k, 4] < 300: continue
        r = cc == k; med = np.median(L[r])
        above = r & (L > med) & (chroma < 40)
        newL[above] = med + (L[above] - med) * K
    w = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.0)
    A[..., 0] = L + (newL - L) * w
    return cv2.cvtColor(np.clip(np.round(A), 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def lab_of(labs, i, W, H):
    l = np.array(Image.open(labs[i])); l = l if l.ndim == 2 else l[..., 0]
    return cv2.resize(l, (W, H), interpolation=cv2.INTER_NEAREST)


if __name__ == '__main__':
    a = sys.argv[1:]; one = None
    if a[0] == '--frame': one, a = int(a[1]), a[2:]
    IN, LABD, OUT = a[:3]
    labs = sorted(glob.glob(os.path.join(LABD, '*.png'))); cap = cv2.VideoCapture(IN); W, H = int(cap.get(3)), int(cap.get(4))
    if one is not None:
        cap.set(1, one); cv2.imwrite(OUT, process(cap.read()[1], lab_of(labs, one, W, H))); sys.exit(0)
    vw = cv2.VideoWriter(OUT, fourcc_for(OUT), cap.get(5) or 30, (W, H)); n = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        vw.write(process(f, lab_of(labs, n, W, H)) if n < len(labs) else f); n += 1
    vw.release(); print(f'wrote {OUT}: {n} frames (car matte, K {K})')

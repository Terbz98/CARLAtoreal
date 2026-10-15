#!/usr/bin/env python3
"""Remove the bright halo just OUTSIDE vehicle silhouettes, against any background.

WHY. User, 2026-10-15: "some cars still have that plasticky white border look" after the road rim was fixed.
Per-vehicle measurement (Town05/Town10HD, v98sev): the outer 0-2 px of some vehicles sit +20..+38 L above the body
3-6 px inside -- a bright outline. CARLA's car materials catch a sky reflection along the silhouette (Fresnel),
and the vehicle pass carries CARLA's shading and mid band, outline included. Real cars at this size do not show it.

Pixels within RIM px inside a vehicle's edge may be no brighter (Lab L) than the local body mean just inside that
band + MARGIN; colour channels untouched. Each connected vehicle region is handled with its own body reference.
  usage: car_edge_derim.py <in.mp4> <label_dir> <out.mp4> | --frame N ...   env: RIM=3 MARGIN=4
"""
import glob, os, sys
import cv2, numpy as np
from PIL import Image

from carla2real.common.vidcodec import fourcc_for
RIM, MARGIN = int(os.environ.get('RIM', '3')), float(os.environ.get('MARGIN', '4'))
VEH = [52, 54, 55, 56, 57, 58, 59, 60, 61, 62]


def process(im, lab):
    m = np.isin(lab, VEH).astype(np.uint8)
    if m.sum() < 500: return im
    # OUTSIDE the vehicle: any non-vehicle, non-sky pixel within RIM px may be no brighter than the same-class
    # surroundings just beyond that band (+MARGIN). The road-shadow HALOFIX only covered ground classes; cars seen
    # against buildings, poles or fences kept a halo (per-car worst +22 L).
    near = (cv2.dilate(m, np.ones((2 * RIM + 1, 2 * RIM + 1), np.uint8)) > 0) & (m == 0) & (lab != 27)
    beyond = (cv2.dilate(m, np.ones((6 * RIM + 3, 6 * RIM + 3), np.uint8)) > 0) & ~near & (m == 0) & (lab != 27)
    A = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32); L = A[..., 0]
    den = cv2.GaussianBlur(beyond.astype(np.float32), (0, 0), 8.0)
    ref = cv2.GaussianBlur(L * beyond, (0, 0), 8.0) / np.maximum(den, 1e-3)
    capL = ref + MARGIN
    fix = near & (den > 1e-2) & (L > capL)
    newL = np.where(fix, capL, L)
    A[..., 0] = L + cv2.GaussianBlur((newL - L).astype(np.float32), (0, 0), 0.8)
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
    vw.release(); print(f'wrote {OUT}: {n} frames (car edge de-rim, RIM {RIM})')

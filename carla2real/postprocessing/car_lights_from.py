#!/usr/bin/env python3
"""Night: take the vehicle lights from another render (v93n), keep everything else.

WHY. User, 2026-10-22: car lights should be "as white as the previous v93n one, like really damn white". L30b is built
on v97n (trained with warm-lamp data), whose car lights render pink/warm; v93n's are pure white. Same drive, same frame
index, so v93n frame t is pixel-aligned. Mask: vehicle pixels (+MARGIN px for the glow) where the DONOR is bright,
w = smoothstep(LO, HI, donor L); feathered; base * (1-w) + donor * w.
  usage: car_lights_from.py <base.mp4> <donor.mp4> <label_dir> <out> | --frame N ...   env: LO=110 HI=190 MARGIN=10 BEAM=0
"""
import glob, os, sys
import cv2, numpy as np
from PIL import Image

from carla2real.common.vidcodec import fourcc_for
LO, HI, M = float(os.environ.get('LO', '110')), float(os.environ.get('HI', '190')), int(os.environ.get('MARGIN', '10'))
EGO = float(os.environ.get('EGO', '0'))      # >0: also the ego headlight pool, lowest EGO fraction of the frame
BODY = os.environ.get('BODY', '0') == '1'   # 1 = old behaviour: beam colour zone also covers vehicle bodies
BEAM = float(os.environ.get('BEAM', '0'))   # >0: also whiten the light thrown below/ahead of each vehicle, BEAM x its height
VEH = [52, 54, 55, 56, 57, 58, 59, 60, 61, 62]


def process(base, donor, l):
    v = np.isin(l, VEH).astype(np.uint8)
    if not v.any(): return base
    zone = cv2.GaussianBlur(cv2.dilate(v, np.ones((2 * M + 1, 2 * M + 1), np.uint8)).astype(np.float32), (0, 0), M / 2)
    L = cv2.cvtColor(donor, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    t = np.clip((L - LO) / (HI - LO), 0, 1)
    w = cv2.GaussianBlur(zone * t * t * (3 - 2 * t), (0, 0), 1.5)[..., None]
    out = np.clip(np.round(base * (1 - w) + donor * w), 0, 255).astype(np.uint8)
    if BEAM <= 0: return out
    # the light the lamps THROW must be white too (user, 2026-10-22): below/ahead of each vehicle, on the ground and
    # nearby surfaces, take colour (a,b) from the donor and keep this render's brightness and texture -> no seams
    n, cc, st, _ = cv2.connectedComponentsWithStats(v)
    bz = np.zeros(v.shape, np.float32); H, W = v.shape
    for i in range(1, n):
        x, y, bw, bh, a = st[i]
        if a < 300: continue
        x0, x1 = max(0, int(x - 0.6 * bw)), min(W, int(x + 1.6 * bw)); y0, y1 = int(y + 0.5 * bh), min(H, int(y + bh + BEAM * bh))
        bz[y0:y1, x0:x1] = 1
    if EGO > 0:   # our own car's headlights light the road ahead of the camera: lower-centre trapezoid of ground
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        top = H * (1 - EGO); half = (W * 0.18) + (yy - top) / max(H - top, 1) * W * 0.45
        bz = np.maximum(bz, ((yy > top) & (np.abs(xx - W / 2) < half) & np.isin(l, [13, 15, 2, 23, 24])).astype(np.float32))
    bz = bz * (~np.isin(l, [27, 17, 6]))   # not onto sky or building fronts far behind
    if BODY: bz = np.maximum(bz, zone)
    else: bz = bz * (1 - v)              # the thrown light only: leave car bodies' colour alone (re-colouring whole
                                         # vehicles from v93n cost night Town04 CIPO .904 -> .856, 2026-10-28)
    bz = cv2.GaussianBlur(bz, (0, 0), 12)[..., None]
    A, Dn = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32), cv2.cvtColor(donor, cv2.COLOR_BGR2LAB).astype(np.float32)
    A[..., 1:] = A[..., 1:] * (1 - bz) + Dn[..., 1:] * bz
    return cv2.cvtColor(np.clip(np.round(A), 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def lab_of(labs, i, W, H):
    l = np.array(Image.open(labs[i])); l = l if l.ndim == 2 else l[..., 0]
    return cv2.resize(l, (W, H), interpolation=cv2.INTER_NEAREST)


if __name__ == '__main__':
    a = sys.argv[1:]; one = None
    if a[0] == '--frame': one, a = int(a[1]), a[2:]
    IN, DON, LABD, OUT = a[:4]
    labs = sorted(glob.glob(os.path.join(LABD, '*.png'))); c1, c2 = cv2.VideoCapture(IN), cv2.VideoCapture(DON)
    W, H = int(c1.get(3)), int(c1.get(4))
    if one is not None:
        c1.set(1, one); c2.set(1, one)
        cv2.imwrite(OUT, process(c1.read()[1], cv2.resize(c2.read()[1], (W, H)), lab_of(labs, one, W, H))); sys.exit(0)
    vw = cv2.VideoWriter(OUT, fourcc_for(OUT), c1.get(5) or 30, (W, H)); n = 0
    while True:
        ok1, f = c1.read(); ok2, g = c2.read()
        if not ok1: break
        if ok2 and n < len(labs): f = process(f, cv2.resize(g, (W, H)), lab_of(labs, n, W, H))
        vw.write(f); n += 1
    vw.release(); print(f'wrote {OUT}: {n} frames (vehicle lights from {os.path.basename(DON)})')

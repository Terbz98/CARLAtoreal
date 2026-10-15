#!/usr/bin/env python3
"""Traffic-sign faces from CARLA: the label map says "sign" but not WHICH sign, so the generator paints a blank.

WHY. User, 2026-10-15: on Town07 "both v98m and v98sev can't detect the sign in the beginning of the clip, the red
triangle sign". The render draws the YIELD sign as a grey triangle -- correct shape, no face. CARLA knows the face.

Per frame, pixels labelled traffic-sign front (50) get CARLA's colour, brightness-matched: CARLA's sign pixels are
scaled in Lab L by the ratio of the render's to CARLA's median lightness in a ring around the sign, so the sign sits
in the render's exposure instead of glowing. Feathered 1.5 px; STRENGTH blends.
  usage: sign_pass.py <in.mp4> <carla_rgb_dir> <label_dir> <out.mp4> | --frame N ... <out.png>   env: STRENGTH=0.9
"""
import glob, os, sys
import cv2, numpy as np
from PIL import Image

from carla2real.common.vidcodec import fourcc_for
S = float(os.environ.get('STRENGTH', '0.9'))
SIGN = [50]


def process(im, car, lab):
    m = np.isin(lab, SIGN).astype(np.uint8)
    if m.sum() < 30: return im
    A = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32); C = cv2.cvtColor(car, cv2.COLOR_BGR2LAB).astype(np.float32)
    ring = (cv2.dilate(m, np.ones((31, 31), np.uint8)) > 0) & (m == 0)
    k = (np.median(A[..., 0][ring]) + 5) / (np.median(C[..., 0][ring]) + 5) if ring.sum() > 50 else 1.0
    k = float(np.clip(k, 0.35, 1.2))
    T = C.copy(); T[..., 0] *= k
    w = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.0)[..., None] * S
    out = A * (1 - w) + T * w
    return cv2.cvtColor(np.clip(np.round(out), 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def load(rgbs, labs, i, W, H):
    car = cv2.resize(np.ascontiguousarray(cv2.imread(rgbs[i])[:, :, ::-1]), (W, H), interpolation=cv2.INTER_AREA)
    l = np.array(Image.open(labs[i])); l = l if l.ndim == 2 else l[..., 0]
    return car, cv2.resize(l, (W, H), interpolation=cv2.INTER_NEAREST)


if __name__ == '__main__':
    a = sys.argv[1:]; one = None
    if a[0] == '--frame': one, a = int(a[1]), a[2:]
    IN, RGBD, LABD, OUT = a[:4]
    rgbs = sorted(glob.glob(os.path.join(RGBD, '*.png'))); labs = sorted(glob.glob(os.path.join(LABD, '*.png')))
    cap = cv2.VideoCapture(IN); W, H = int(cap.get(3)), int(cap.get(4))
    if one is not None:
        cap.set(1, one); cv2.imwrite(OUT, process(cap.read()[1], *load(rgbs, labs, one, W, H))); sys.exit(0)
    vw = cv2.VideoWriter(OUT, fourcc_for(OUT), cap.get(5) or 30, (W, H)); n = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        vw.write(process(f, *load(rgbs, labs, n, W, H)) if n < min(len(rgbs), len(labs)) else f); n += 1
    vw.release(); print(f'wrote {OUT}: {n} frames (sign faces from CARLA)')

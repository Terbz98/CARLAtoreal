#!/usr/bin/env python3
"""Soften vegetation: remove the over-sharp, cartoonish micro-contrast on trees.

WHY. User, 2026-10-08, on sunny baseline v90dv: "the trees still look cartoonish ... all ur fixes made it
even worse and too sharp ... not real at all". Real foliage at driving distance is a soft, low-contrast
texture; the generator plus the delivery chain's sharpening draw crisp high-contrast leaf edges.

On vegetation pixels (label 30, feathered; terrain 29 at half weight): the fine band (below SIGMA_F px)
is scaled by FINE, and local contrast around a SIGMA_M px mean by MID. Vehicles, poles and sky untouched.
  usage: tree_soften.py <in_video> <label_dir> <out_video>  |  --frame N <in_video> <label_dir> <out.png>
  env:   FINE=0.45 MID=0.8 SIGMA_F=1.2 SIGMA_M=6
"""
import glob, os, sys
import cv2, numpy as np
from PIL import Image

from carla2real.common.vidcodec import fourcc_for
FINE, MID = float(os.environ.get('FINE', '0.45')), float(os.environ.get('MID', '0.8'))
SF, SM = float(os.environ.get('SIGMA_F', '1.2')), float(os.environ.get('SIGMA_M', '6'))


def process(im, lab):
    H, W = im.shape[:2]
    lab = cv2.resize(lab, (W, H), interpolation=cv2.INTER_NEAREST)
    m = (lab == 30).astype(np.float32) + 0.5 * (lab == 29)
    m = cv2.GaussianBlur(cv2.erode(m, np.ones((3, 3), np.uint8)), (0, 0), 2.0)[..., None]
    x = im.astype(np.float32)
    lo_f = cv2.GaussianBlur(x, (0, 0), SF)
    lo_m = cv2.GaussianBlur(x, (0, 0), SM)
    soft = lo_m + MID * (lo_f - lo_m) + FINE * (x - lo_f)
    return np.clip(x * (1 - m) + soft * m, 0, 255).astype(np.uint8)


def label(labs, i):
    l = np.array(Image.open(labs[i])); return l if l.ndim == 2 else l[..., 0]


if __name__ == '__main__':
    a = sys.argv[1:]
    one = None
    if a[0] == '--frame': one, a = int(a[1]), a[2:]
    IN, LABD, OUT = a[:3]
    labs = sorted(glob.glob(os.path.join(LABD, '*.png')))
    cap = cv2.VideoCapture(IN); W, H = int(cap.get(3)), int(cap.get(4))
    if one is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, one); cv2.imwrite(OUT, process(cap.read()[1], label(labs, one))); sys.exit(0)
    vw = cv2.VideoWriter(OUT, fourcc_for(OUT), cap.get(5) or 30, (W, H)); n = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        vw.write(process(f, label(labs, n)) if n < len(labs) else f); n += 1
    vw.release(); print(f'wrote {OUT}: {n} frames (fine {FINE}, mid {MID})')

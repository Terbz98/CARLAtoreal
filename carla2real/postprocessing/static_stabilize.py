#!/usr/bin/env python3
"""Temporal stabilisation of STATIC surfaces, with motion taken from CARLA's own (flicker-free) frames.

WHY. User, 2026-10-15, after far_stabilize.py: "for v98f the buildings still flicker a shit ton". Low-passed temporal
change after motion compensation (motion from CARLA), Town10HD f300-340: buildings v98f 5.6 vs CARLA 1.8 (3.2x),
vegetation 6.3 vs 2.1 -- across the whole facade band, not only far away. far_stabilize.py only touched far buildings,
and estimated motion on the render itself, whose flicker corrupts the flow.

Per frame t: dense flow from CARLA frame t to t-1 (Farneback, half size) -- the true camera/scene motion; the previous
OUTPUT is warped onto t and blended in with weight ALPHA on static classes (building, wall, fence, guard rail, bridge,
vegetation, terrain, pole, sidewalk), gated by agreement (|dL| < AGREE) so disocclusions and moving objects never smear.
Vehicles, people, sky, road (already stable) untouched. Recursive: effective window ~1/(1-ALPHA) frames.
  usage: static_stabilize.py <in.mp4> <carla_rgb_dir> <label_dir> <out.mp4>   env: ALPHA=0.8 AGREE=30
"""
import glob, os, sys
import cv2, numpy as np
from PIL import Image

from carla2real.common.vidcodec import fourcc_for
AL, AG = float(os.environ.get('ALPHA', '0.8')), float(os.environ.get('AGREE', '30'))
CLS = [17, 6, 3, 4, 16, 30, 29, 45, 15]
IN, RGBD, LABD, OUT = sys.argv[1:5]
rgbs = sorted(glob.glob(os.path.join(RGBD, '*.png'))); labs = sorted(glob.glob(os.path.join(LABD, '*.png')))
cap = cv2.VideoCapture(IN); W, H = int(cap.get(3)), int(cap.get(4))
vw = cv2.VideoWriter(OUT, fourcc_for(OUT), cap.get(5) or 30, (W, H))
gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
cg = lambda i: cv2.cvtColor(cv2.resize(cv2.imread(rgbs[i]), (W // 2, H // 2), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
prev = None; pg = None; n = 0
while True:
    ok, f = cap.read()
    if not ok: break
    out = f
    if n < len(rgbs) and n < len(labs):
        g = cg(n)
        if prev is not None:
            l = np.array(Image.open(labs[n])); l = l if l.ndim == 2 else l[..., 0]
            st = cv2.resize(np.isin(l, CLS).astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
            fl = cv2.resize(cv2.calcOpticalFlowFarneback(g, pg, None, 0.5, 4, 21, 3, 5, 1.2, 0), (W, H)) * 2.0
            warped = cv2.remap(prev, gx + fl[..., 0], gy + fl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            dL = np.abs(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(np.float32) - cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32))
            agree = np.clip(1 - (dL - AG / 2) / (AG / 2), 0, 1)
            w = cv2.GaussianBlur(cv2.erode(st, np.ones((3, 3), np.uint8)) * agree, (0, 0), 1.5)[..., None] * AL
            out = np.clip(f.astype(np.float32) * (1 - w) + warped.astype(np.float32) * w, 0, 255).astype(np.uint8)
        pg = g
    vw.write(out); prev = out; n += 1
vw.release(); print(f'wrote {OUT}: {n} frames (static stabilise, alpha {AL}, CARLA flow)')

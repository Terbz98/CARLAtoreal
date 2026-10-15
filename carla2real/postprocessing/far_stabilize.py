#!/usr/bin/env python3
"""Temporal stabilisation of DISTANT buildings: kills shimmer on far facades without touching anything near.

WHY. User, 2026-10-15: "the buildings from afar in town 10 are so damn flickery". Far facades are a few pixels of
fine texture that the generator re-invents every frame; up close the same texture is stable because it is large.

Per frame: optical flow (DIS, on a half-size frame) from the current frame to the previous OUTPUT; the previous
output is warped onto the current frame and blended in with weight ALPHA where (a) the label is building/wall/bridge
(17, 6, 16), (b) the CARLA depth map says far (depth < FAR_MAX on its 255=near scale, ramped), and (c) the warped
pixel agrees with the current one (|dL| < AGREE) -- disocclusions and moving objects are never smeared. Recursive,
so the effective window is ~1/(1-ALPHA) frames.
  usage: far_stabilize.py <in.mp4> <label_dir> <depth_dir> <out.mp4>   env: ALPHA=0.7 FAR_MAX=110 AGREE=28
"""
import glob, os, sys
import cv2, numpy as np
from PIL import Image

from carla2real.common.vidcodec import fourcc_for
AL, FM, AG = float(os.environ.get('ALPHA', '0.7')), float(os.environ.get('FAR_MAX', '110')), float(os.environ.get('AGREE', '28'))
CLS = [17, 6, 16]
IN, LABD, DEPD, OUT = sys.argv[1:5]
labs = sorted(glob.glob(os.path.join(LABD, '*.png'))); deps = sorted(glob.glob(os.path.join(DEPD, '*.png')))
cap = cv2.VideoCapture(IN); W, H = int(cap.get(3)), int(cap.get(4))
vw = cv2.VideoWriter(OUT, fourcc_for(OUT), cap.get(5) or 30, (W, H))
dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
prev = None; n = 0
gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
while True:
    ok, f = cap.read()
    if not ok: break
    out = f
    if prev is not None and n < len(labs) and n < len(deps):
        l = np.array(Image.open(labs[n])); l = l if l.ndim == 2 else l[..., 0]
        l = cv2.resize(l, (W, H), interpolation=cv2.INTER_NEAREST)
        d = cv2.resize(cv2.imread(deps[n], cv2.IMREAD_GRAYSCALE), (W, H)).astype(np.float32)
        far = np.clip((FM - d) / 40.0, 0, 1) * np.isin(l, CLS)
        if far.sum() > 500:
            a = cv2.cvtColor(cv2.resize(f, (W // 2, H // 2)), cv2.COLOR_BGR2GRAY)
            b = cv2.cvtColor(cv2.resize(prev, (W // 2, H // 2)), cv2.COLOR_BGR2GRAY)
            fl = cv2.resize(dis.calc(a, b, None), (W, H)) * 2.0
            warped = cv2.remap(prev, gx + fl[..., 0], gy + fl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            dL = np.abs(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(np.float32) - cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32))
            agree = np.clip(1 - (dL - AG / 2) / (AG / 2), 0, 1)
            w = cv2.GaussianBlur((far * agree).astype(np.float32), (0, 0), 1.5)[..., None] * AL
            out = np.clip(f.astype(np.float32) * (1 - w) + warped.astype(np.float32) * w, 0, 255).astype(np.uint8)
    vw.write(out); prev = out; n += 1
vw.release(); print(f'wrote {OUT}: {n} frames (far-building stabilise, alpha {AL})')

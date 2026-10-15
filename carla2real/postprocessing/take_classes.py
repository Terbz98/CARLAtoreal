#!/usr/bin/env python3
"""Take selected classes from a second render of the same frames (feathered), e.g. fences from v98m into v98sev.
  usage: take_classes.py <base.mp4> <donor.mp4> <label_dir> <out.mp4> <id,id,...>
"""
import glob, os, sys
import cv2, numpy as np
from PIL import Image

from carla2real.common.vidcodec import fourcc_for
BASE, DON, LABD, OUT, IDS = sys.argv[1:6]; IDS = [int(x) for x in IDS.split(',')]
labs = sorted(glob.glob(os.path.join(LABD, '*.png'))); a, b = cv2.VideoCapture(BASE), cv2.VideoCapture(DON)
W, H = int(a.get(3)), int(a.get(4)); vw = cv2.VideoWriter(OUT, fourcc_for(OUT), a.get(5) or 30, (W, H)); n = 0
while True:
    ok1, f = a.read(); ok2, g = b.read()
    if not (ok1 and ok2): break
    if n < len(labs):
        l = np.array(Image.open(labs[n])); l = l if l.ndim == 2 else l[..., 0]
        m = cv2.GaussianBlur(cv2.resize(np.isin(l, IDS).astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST), (0, 0), 1.2)[..., None]
        f = np.clip(f * (1 - m) + cv2.resize(g, (W, H)) * m, 0, 255).astype(np.uint8)
    vw.write(f); n += 1
vw.release(); print(f'wrote {OUT}: {n} frames, classes {IDS} from {os.path.basename(DON)}')

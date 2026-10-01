#!/usr/bin/env python3
"""Aerial perspective: distant things lose colour and pick up haze. The renders do not.

WHY. The user, on v90dv (2026-10-01): "the far objects like trees and stuff are too vibrant n vivid,
looks too fake". Measured, vegetation saturation by depth band, against real photographs whose depth
maps come from the same estimator the corpus uses:

                         near    mid    far
    real Mapillary      104.5   86.7   68.3
    v50m                117.4   90.4   39.5
    v90dv               134.6  118.8   69.2

The tree lines down a street sit in the MID band of this inverse-depth encoding, and there v90dv
runs 37% above real photographs. The generator has no notion of atmosphere; real cameras always see
some, and the eye reads its absence as "CG" before it can say why.

WHAT IT DOES, per pixel, driven by the per-frame depth map the generator was conditioned on (so it is
stable frame to frame and needs no temporal filter):
  * vegetation and terrain: saturation scaled from SAT_NEAR (closest) to SAT_FAR (farthest)
  * everything that is not sky and not a vehicle: blended toward the frame's own sky colour by up to
    HAZE at the far end, with the weight rising as farness^HAZE_GAMMA so the near field is untouched
Vehicles are excluded entirely -- the CARLA vehicle pass owns them and the user likes them.

Depth maps here are 255 = nearest, 0 = farthest.

  usage: aerial_pass.py <in_video> <label_dir> <depth_dir> <out_video>
         aerial_pass.py --frame N <in_video> <label_dir> <depth_dir> <out.png>
  env:   SAT_NEAR=0.85 SAT_FAR=0.55 HAZE=0.18 HAZE_GAMMA=1.6
"""
import glob
import os
import sys

import cv2
import numpy as np
from PIL import Image

from carla2real.common.vidcodec import fourcc_for

SAT_NEAR = float(os.environ.get('SAT_NEAR', '0.85'))
SAT_FAR = float(os.environ.get('SAT_FAR', '0.55'))
HAZE = float(os.environ.get('HAZE', '0.18'))
HAZE_GAMMA = float(os.environ.get('HAZE_GAMMA', '1.6'))

VEG = [30, 29]                          # Mapillary-65 vegetation, terrain
SKY = [27]
VEHICLES = [54, 55, 56, 57, 58, 61, 52, 53, 57]


def process(im, lab, dep):
    H, W = im.shape[:2]
    lab = cv2.resize(lab, (W, H), interpolation=cv2.INTER_NEAREST)
    far = 1.0 - cv2.GaussianBlur(cv2.resize(dep, (W, H)).astype(np.float32), (0, 0), 3) / 255.0
    far = np.clip(far, 0, 1)
    sky = np.isin(lab, SKY)
    veh = cv2.dilate(np.isin(lab, VEHICLES).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    veg = cv2.GaussianBlur(np.isin(lab, VEG).astype(np.float32), (0, 0), 2)

    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype(np.float32)
    mult = SAT_NEAR + (SAT_FAR - SAT_NEAR) * far
    hsv[..., 1] *= 1 - veg * (1 - mult)
    out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    # haze toward this frame's own sky colour; fall back to a pale neutral when no sky is visible
    hz = im[sky].mean(0) if sky.sum() > 2000 else np.array([200, 200, 200], np.float32)
    w = HAZE * far ** HAZE_GAMMA
    w[sky] = 0
    w = cv2.GaussianBlur(w * (~veh), (0, 0), 2)[..., None]
    out = out * (1 - w) + hz[None, None, :] * w
    keep = cv2.GaussianBlur(veh.astype(np.float32), (0, 0), 2)[..., None]
    out = out * (1 - keep) + im.astype(np.float32) * keep
    return np.clip(out, 0, 255).astype(np.uint8)


def read_pair(labs, deps, i):
    lab = np.array(Image.open(labs[i]))
    lab = lab if lab.ndim == 2 else lab[..., 0]
    return lab, cv2.imread(deps[i], cv2.IMREAD_GRAYSCALE)


if __name__ == '__main__':
    a = sys.argv[1:]
    one = None
    if a[0] == '--frame':
        one, a = int(a[1]), a[2:]
    IN, LABD, DEPD, OUT = a[:4]
    labs = sorted(glob.glob(os.path.join(LABD, '*.png')))
    deps = sorted(glob.glob(os.path.join(DEPD, '*.png')))
    cap = cv2.VideoCapture(IN)
    W, H = int(cap.get(3)), int(cap.get(4))
    if one is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, one)
        im = cap.read()[1]
        lab, dep = read_pair(labs, deps, one)
        cv2.imwrite(OUT, process(im, lab, dep))
        sys.exit(0)
    vw = cv2.VideoWriter(OUT, fourcc_for(OUT), cap.get(5) or 30, (W, H))
    n = 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if n < len(labs) and n < len(deps):
            lab, dep = read_pair(labs, deps, n)
            im = process(im, lab, dep)
        vw.write(im)
        n += 1
    vw.release()
    print(f'wrote {OUT}: {n} frames  (sat {SAT_NEAR}->{SAT_FAR}, haze {HAZE})')

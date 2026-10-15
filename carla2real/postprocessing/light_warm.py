#!/usr/bin/env python3
"""Night: warm (sodium) tint only where CARLA's lamps actually light the scene -- the v51 look.

WHY. User, 2026-10-15: "for night, can we just follow how v51 does the yellowing, like only on the lights part".
v51's amber sits on lamp-lit surfaces (car roofs, lit windows, the road pool under a lamp), not on the whole
frame; a global warm cast tints the dark too. The night model is conditioned on a LIGHT map (test_*_night_inst_gt
_light: CARLA's own illumination), which marks exactly those surfaces.

w = (light/255) ** GAMMA, feathered; b += WARM_B * w, a += WARM_A * w; already-coloured pixels (traffic lights,
tail lights) gated out by render chroma.
  usage: light_warm.py <in.mp4> <light_dir> <out.mp4> [label_dir] | --frame N <in.mp4> <light_dir> <out.png> [label_dir]
  env:   WARM_B=40 WARM_A=10 GAMMA=0.5 CHROMA_MAX=60
"""
import glob, os, sys
import cv2, numpy as np

from carla2real.common.vidcodec import fourcc_for
WB, WA = float(os.environ.get('WARM_B', '40')), float(os.environ.get('WARM_A', '10'))
G, CM = float(os.environ.get('GAMMA', '0.5')), float(os.environ.get('CHROMA_MAX', '60'))


def process(im, li, lab_ids=None):
    H, W = im.shape[:2]
    li = cv2.resize(li if li.ndim == 2 else li[..., 0], (W, H)).astype(np.float32) / 255.0
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
    w = cv2.GaussianBlur(li ** G, (0, 0), 3.0)
    chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    w *= np.clip(1 - chroma / CM, 0, 1)
    if lab_ids is not None:   # lane markings stay white: yellow paint means something else to a driver
        mk = cv2.resize(np.isin(lab_ids, [23, 24]).astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
        w *= 1 - cv2.GaussianBlur(cv2.dilate(mk, np.ones((5, 5), np.uint8)), (0, 0), 1.5)
        # vehicles and their own lights stay neutral: user, 2026-10-15, "the vehicles' lights shouldn't be yellow,
        # only the street lamps". Vehicle bodies, head/tail lights and their glow (VEH_MARGIN px) are excluded.
        vm = cv2.resize(np.isin(lab_ids, [52, 54, 55, 56, 57, 58, 59, 60, 61, 62]).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        VM = int(os.environ.get('VEH_MARGIN', '25'))
        w *= 1 - cv2.GaussianBlur(cv2.dilate(vm, np.ones((VM, VM), np.uint8)).astype(np.float32), (0, 0), VM / 3)
    lab[..., 2] += WB * w; lab[..., 1] += WA * w
    return cv2.cvtColor(np.clip(np.round(lab), 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


if __name__ == '__main__':
    a = sys.argv[1:]; one = None
    if a[0] == '--frame': one, a = int(a[1]), a[2:]
    IN, LD, OUT = a[:3]
    lbs = sorted(glob.glob(os.path.join(a[3], '*.png'))) if len(a) > 3 else []
    lid = lambda i: (lambda x: x if x.ndim == 2 else x[..., 0])(cv2.imread(lbs[i], -1)) if i < len(lbs) else None
    lis = sorted(glob.glob(os.path.join(LD, '*.png'))); cap = cv2.VideoCapture(IN)
    if one is not None:
        cap.set(1, one); cv2.imwrite(OUT, process(cap.read()[1], cv2.imread(lis[one], -1), lid(one))); sys.exit(0)
    W, H = int(cap.get(3)), int(cap.get(4)); vw = cv2.VideoWriter(OUT, fourcc_for(OUT), cap.get(5) or 30, (W, H)); n = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        vw.write(process(f, cv2.imread(lis[n], -1), lid(n)) if n < len(lis) else f); n += 1
    vw.release(); print(f'wrote {OUT}: {n} frames (warm b {WB}, a {WA})')

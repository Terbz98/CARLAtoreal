#!/usr/bin/env python3
"""Give each vehicle back what only CARLA knows about it: its shading, its structure and its livery.

WHY A POST-PASS AND NOT MORE TRAINING. The user's worst complaint is the Town05 ambulance, and the
thing wrong with it is information the generator has never had. The label map says "truck" and
nothing more. No licensed photograph -- no photograph at all -- shows THIS CARLA ambulance's
chevrons, its bumper step, its light bar. v50m did not know them either; it just painted warmer and
bolder. CARLA renders them every frame, from the same camera, pixel-aligned with the label map. This
is the move protect_buildings.py made for facades (+21% facade detail, and the shimmer gone), moved
to vehicles, where it was never applied.

protect_vehicles.py tried only the chroma third of this and the gain was marginal by eye, because
the defects that read worst are in LIGHTNESS, not colour:

  * the Town05 jeep is pale grey at the front and maroon at the back -- one car in two tones. CARLA
    shades it as one body lit from one side. That is low-frequency lightness.
  * the ambulance's lower half is a blank white slab where CARLA has a bumper, a step, tail lights
    and a dark underbody. That is mid-frequency lightness structure.

THREE BANDS, per vehicle, in Lab, pixel-aligned with CARLA:

  shading    L below SIGMA_LO. The render's own large-scale shading is replaced by CARLA's, scaled
             by SHADE_GAIN, around the render's own mean. CARLA decides which panel is lit; the render
             keeps its exposure. CARLA is the reference, not zero: v50m sits just ABOVE CARLA on this
             measure (split 39 vs 35), v85d below (33), v86d far below (22), and the user ranks them
             in exactly that order.
  structure  L between SIGMA_HI and SIGMA_LO. CARLA's panel lines, bumper, windows and lights,
             ADDED to the render's, weighted by STRUCT_GAIN. Added, not swapped, and only in this
             band: the render's finest texture above SIGMA_HI -- paint grain, reflections, the
             photoreal part -- is never touched, which is what stops the result reading as CG.
  livery     a/b below SIGMA_LO, the same two-way match as protect_vehicles.py: CARLA's colour
             structure where it has more (markings), the render's damped where it invented some.

GUARDS. Per object, never per frame. Masks eroded so silhouettes and their blend with the road are
untouched. Mask-normalised blurs, so the background never bleeds into a vehicle's statistics. Small
vehicles (< MIN_PX) are left alone: at that size CARLA's structure is aliasing, not information.

  usage: vehicle_pass.py <in_video> <carla_rgb_dir> <label_dir> <out_video>
         vehicle_pass.py --frame N <in_video> <carla_rgb_dir> <label_dir> <out.png>   (one frame)
  env:   SHADE_GAIN=1.0 STRUCT_GAIN=0.6 LIVERY=1.0 SIGMA_LO=6 SIGMA_HI=1.2 SIGMA_C=2.0 MIN_PX=1500
"""
import glob
import os
import sys

import cv2
import numpy as np
from PIL import Image

from carla2real.common.vidcodec import fourcc_for

SHADE_GAIN = float(os.environ.get('SHADE_GAIN', '1.0'))
STRUCT_GAIN = float(os.environ.get('STRUCT_GAIN', '1.0'))
LIVERY = float(os.environ.get('LIVERY', '1.0'))
SIGMA_LO = float(os.environ.get('SIGMA_LO', '6'))
SIGMA_HI = float(os.environ.get('SIGMA_HI', '1.2'))
# livery lives at a finer scale than shading: a chevron stripe is ~8 px wide at 1920, so blurring
# CARLA's colour at SIGMA_LO smeared the ambulance's markings into a watercolour wash
SIGMA_C = float(os.environ.get('SIGMA_C', '2.0'))
MIN_PX = int(os.environ.get('MIN_PX', '1500'))
GLASS_L = float(os.environ.get('GLASS_L', '70'))
BODY = float(os.environ.get('BODY', '0.6'))        # 0 = render's body colour, 1 = CARLA's   # Lab L (0-255) below which CARLA pixels are glass/tyre
FEATHER = 2.5
VEHICLES = [54, 55, 56, 57, 58, 61]          # Mapillary-65 vehicle bodies


def mblur(x, m, s):
    """Gaussian blur restricted to the mask, so the road never leaks into a car's statistics."""
    num = cv2.GaussianBlur(x * m, (0, 0), s)
    den = cv2.GaussianBlur(m, (0, 0), s)
    return num / np.maximum(den, 1e-4)


def process(im, car, lab):
    H, W = im.shape[:2]
    m_all = np.isin(lab, VEHICLES).astype(np.uint8)
    if m_all.sum() < MIN_PX:
        return im, 0
    m_all = cv2.erode(m_all, np.ones((3, 3), np.uint8), iterations=2)
    R = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
    C = cv2.cvtColor(car, cv2.COLOR_BGR2LAB).astype(np.float32)
    out = R.copy()
    acc = np.zeros((H, W), np.float32)
    ncc, cc, st, _ = cv2.connectedComponentsWithStats(m_all, 8)
    touched = 0
    for ci in range(1, ncc):
        if st[ci, 4] < MIN_PX:
            continue
        x, y, w, h = st[ci, :4]
        p = int(3 * SIGMA_LO)
        x0, y0, x1, y1 = max(0, x - p), max(0, y - p), min(W, x + w + p), min(H, y + h + p)
        reg = (cc[y0:y1, x0:x1] == ci)
        m = reg.astype(np.float32)
        r, c = R[y0:y1, x0:x1], C[y0:y1, x0:x1]
        o = out[y0:y1, x0:x1]

        # ---- lightness: shading + structure, render's fine texture untouched
        rL, cL = r[..., 0], c[..., 0]
        r_lo, c_lo = mblur(rL, m, SIGMA_LO), mblur(cL, m, SIGMA_LO)
        r_mid = mblur(rL, m, SIGMA_HI) - r_lo
        c_mid = mblur(cL, m, SIGMA_HI) - c_lo
        r_fine = rL - mblur(rL, m, SIGMA_HI)
        rmean, cmean = rL[reg].mean(), c_lo[reg].mean()
        shading = rmean + SHADE_GAIN * (c_lo - cmean)
        newL = shading + r_mid + STRUCT_GAIN * c_mid + r_fine
        o[..., 0] = np.where(reg, newL, o[..., 0])

        # ---- livery: match chroma structure both ways, keep the render's mean body colour.
        # PAINT vs GLASS/RUBBER. CARLA tints its glass with a reflection -- the Town05 bus windshield
        # is green -- and transferring that painted the render's windshield bright green. Simply
        # leaving dark parts to the render was worse: the jeep's dark spare-wheel cover kept the
        # render's invented maroon. Glass, tyres and trim are close to colourless in reality, so
        # dark CARLA pixels are pulled toward NEUTRAL, keeping only a hint of CARLA's tint.
        paint = np.clip((cv2.GaussianBlur(cL, (0, 0), 1.5) - GLASS_L) / 40.0, 0, 1)
        for k in (1, 2):
            rk, ck = r[..., k], c[..., k]
            rk_lo, ck_lo = mblur(rk, m, SIGMA_C), mblur(ck, m, SIGMA_C)
            r_dev = rk - rk_lo                                   # render's own colour detail
            # the car's overall colour, pulled toward CARLA's by BODY. Keeping only the render's mean
            # left the charcoal Town05 jeep pale MINT: a near-neutral car keeps the render's small tint
            # and the brighter shading amplifies it. CARLA is authoritative about paint colour.
            base = (1 - BODY) * rk[reg].mean() + BODY * ck[reg].mean()
            body = base + (ck_lo - ck[reg].mean())                # plus CARLA's livery structure
            dark = 128.0 + 0.3 * (ck_lo - 128.0)                  # near-neutral glass / rubber
            target = paint * body + (1 - paint) * dark
            newk = (1 - LIVERY) * rk_lo + LIVERY * target + r_dev
            o[..., k] = np.where(reg, newk, o[..., k])
        acc[y0:y1, x0:x1] = np.maximum(acc[y0:y1, x0:x1], m)
        touched += 1

    if not touched:
        return im, 0
    out[..., 0] = np.clip(out[..., 0], 0, 255)
    out[..., 1:] = np.clip(out[..., 1:], 0, 255)
    rec = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
    soft = cv2.GaussianBlur(acc, (0, 0), FEATHER)[..., None]
    return np.clip(im * (1 - soft) + rec * soft, 0, 255).astype(np.uint8), touched


def load_ref(rgbs, labs, i, W, H):
    car = cv2.imread(rgbs[i])
    car = cv2.resize(np.ascontiguousarray(car[:, :, ::-1]), (W, H), interpolation=cv2.INTER_AREA)
    lab = np.array(Image.open(labs[i]))
    lab = lab if lab.ndim == 2 else lab[..., 0]
    return car, cv2.resize(lab, (W, H), interpolation=cv2.INTER_NEAREST)


if __name__ == '__main__':
    a = sys.argv[1:]
    one = None
    if a[0] == '--frame':
        one, a = int(a[1]), a[2:]
    IN, RGB, LAB, OUT = a[:4]
    rgbs = sorted(glob.glob(os.path.join(RGB, '*.png')))
    labs = sorted(glob.glob(os.path.join(LAB, '*.png')))
    cap = cv2.VideoCapture(IN)
    W, H = int(cap.get(3)), int(cap.get(4))
    if one is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, one)
        ok, im = cap.read()
        car, lab = load_ref(rgbs, labs, one, W, H)
        res, t = process(im, car, lab)
        cv2.imwrite(OUT, res)
        print(f'frame {one}: {t} vehicles')
        sys.exit(0)
    vw = cv2.VideoWriter(OUT, fourcc_for(OUT), cap.get(5) or 30, (W, H))
    n = tot = 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if n < len(rgbs) and n < len(labs):
            car, lab = load_ref(rgbs, labs, n, W, H)
            im, t = process(im, car, lab)
            tot += t
        vw.write(im)
        n += 1
        if n % 200 == 0:
            print(f'  {n}', flush=True)
    vw.release()
    print(f'wrote {OUT}: {n} frames, {tot} vehicle-instances processed')

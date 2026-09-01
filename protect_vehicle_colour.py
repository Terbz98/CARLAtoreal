#!/usr/bin/env python3
"""Correct vehicle body COLOUR from CARLA, keeping the render's own texture and lighting.

WHY. The generator repaints vehicles at will. On Town10HD sunny frame 700 CARLA has a beige
Lincoln; v50 renders it dark maroon, and because that car fills half the frame the whole shot
reads as a warm "sandy" cast (R-B +25 against raw CARLA's -6). Measured over the clip, 21-23% of
frames run warm by more than 10 units in BOTH v49 and v50, while raw CARLA never exceeds +0.8 --
so this is the pipeline inventing colour, not the scene.

The chroma channel was supposed to prevent exactly this and does not: at that frame the prior
says the car is near-neutral (R-B -6.1) and the model paints it maroon regardless. A neutral-grey
prior even scores BETTER on colour uniformity than the real one, so the model is not really
following it.

WHAT THIS DOES. Same contract as protect_lane_markings / protect_traffic_lights: take from CARLA
only the thing CARLA is authoritative about. Here that is HUE and SATURATION of the vehicle body.
Luminance -- shading, highlights, reflections, the whole photoreal look -- stays entirely the
render's. So the car keeps its realistic material and gets its correct colour.

Guards, because a body-colour paste is easy to get wrong:
  * only pixels whose label is a vehicle class, eroded so the outline is untouched
  * skips glass and tyres via a luminance band, which are not body paint
  * blends with a soft mask so there is no cut edge
  * per-region: if CARLA and the render already agree on hue, nothing is changed

Usage: protect_vehicle_colour.py in.mp4 carla_rgb_dir label_dir out.mp4 [strength=0.85]
CARLA PNGs are BGR-as-RGB, so channels are swapped on read (see carla_rgb_channel_swap).
"""
import sys, os, glob
import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vidcodec import fourcc_for

IN, RGB_DIR, LBL_DIR, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
STRENGTH = float(sys.argv[5]) if len(sys.argv) > 5 else 0.85
# minimum hue disagreement (OpenCV units, 0-180) before a vehicle is repainted
HUE_TOL = float(os.environ.get('HUE_TOL', '20'))
# below this CARLA-side saturation the region is effectively grey and its hue is noise
MIN_SAT = float(os.environ.get('MIN_SAT', '40'))

# Mapillary-65 vehicle bodies. Riders/persons are deliberately excluded -- skin and clothing are
# not paint and CARLA's are not a colour reference worth transferring.
VEHICLES = [54, 55, 56, 57, 58, 61]

rgbs = sorted(glob.glob(RGB_DIR + '/*.png'))
labs = sorted(glob.glob(LBL_DIR + '/*.png'))
cap = cv2.VideoCapture(IN)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
W, H = int(cap.get(3)), int(cap.get(4))
vw = cv2.VideoWriter(OUT, fourcc_for(OUT), fps, (W, H))

n = fixed = 0
shift_before, shift_after = [], []
while True:
    ok, im = cap.read()
    if not ok:
        break
    if n < len(rgbs) and n < len(labs):
        car = cv2.imread(rgbs[n])
        lab = np.array(Image.open(labs[n]))
        if car is not None:
            car = np.ascontiguousarray(car[:, :, ::-1])           # BGR-as-RGB -> real BGR
            car = cv2.resize(car, (W, H), interpolation=cv2.INTER_AREA)
            lab = lab if lab.ndim == 2 else lab[:, :, 0]
            lab = cv2.resize(lab, (W, H), interpolation=cv2.INTER_NEAREST)

            m = np.isin(lab, VEHICLES).astype(np.uint8)
            if m.sum() > 500:
                # pull the mask in so the silhouette and its blend with the background survive
                m = cv2.erode(m, np.ones((5, 5), np.uint8), iterations=2)
                y = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                # body paint only: drop near-black (tyres, shadow, glass interior) and blown
                # highlights (specular, glazing) -- neither carries usable body hue
                m = (m > 0) & (y > 35) & (y < 235)
                if m.sum() > 500:
                    ih = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype(np.float32)
                    ch = cv2.cvtColor(car, cv2.COLOR_BGR2HSV).astype(np.float32)
                    b0, g0, r0 = [im[:, :, k][m].mean() for k in range(3)]
                    shift_before.append(r0 - b0)
                    # PER REGION, not per pixel. A per-pixel hue paste blotches badly wherever
                    # the render's geometry differs from CARLA's by even a few pixels -- it
                    # turned a white bus into red patches. One median hue per connected vehicle
                    # is what the body actually has, and it is immune to that misalignment.
                    out = ih.copy()
                    ncc, cc = cv2.connectedComponents(m.astype(np.uint8))
                    for ci in range(1, ncc):
                        reg = cc == ci
                        if reg.sum() < 400:
                            continue
                        # circular median hue: OpenCV hue wraps at 180, so average as angles
                        hh = ch[..., 0][reg] * (np.pi / 90.0)
                        mh = (np.arctan2(np.sin(hh).mean(), np.cos(hh).mean()) % (2 * np.pi)) * (90.0 / np.pi)
                        ms = float(np.median(ch[..., 1][reg]))
                        # Only act where the render and CARLA genuinely disagree. Most vehicles
                        # are rendered fine; touching those risks making them worse for no gain
                        # (a per-pixel version of this turned a white bus into red patches).
                        rh = ih[..., 0][reg] * (np.pi / 90.0)
                        rmh = (np.arctan2(np.sin(rh).mean(), np.cos(rh).mean()) % (2 * np.pi)) * (90.0 / np.pi)
                        d = abs(mh - rmh); d = min(d, 180.0 - d)      # hue wraps at 180 in OpenCV
                        if d < HUE_TOL:
                            continue
                        # ACHROMATIC GUARD. Hue is meaningless on a near-neutral surface: for a
                        # grey, cream or silver body the angle is decided by tiny channel
                        # differences, and the circular mean reports it with high confidence
                        # anyway. Measured on Town10HD frame 450, CARLA's vehicles have median
                        # saturation 5-31 out of 255 and yield a consistent "hue" around 125
                        # (blue). Imposing that on the render's red bus and then blending at
                        # STRENGTH gave red-toward-blue = MAGENTA, which is what shipped in v50b
                        # and v50d and is the reported defect.
                        #
                        # When CARLA is neutral the correct correction is not a hue at all -- it
                        # is to pull the render's SATURATION down toward neutral, which removes
                        # the invented colour cast (the whole point of this stage) without
                        # asserting a colour that is not there. So only adopt the hue when CARLA
                        # is chromatic enough for it to mean something.
                        if ms >= MIN_SAT:
                            out[..., 0][reg] = mh
                        # keep the render's own saturation VARIATION, just re-centre its level
                        cur = ih[..., 1][reg]
                        cs = float(np.median(cur)) + 1e-3
                        out[..., 1][reg] = np.clip(cur * (ms / cs), 0, 255)
                    rec = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
                    soft = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 3.0)[..., None]
                    w = soft * STRENGTH
                    im = np.clip(im.astype(np.float32) * (1 - w) + rec * w, 0, 255).astype(np.uint8)
                    b1, g1, r1 = [im[:, :, k][m].mean() for k in range(3)]
                    shift_after.append(r1 - b1)
                    fixed += 1
    vw.write(im)
    n += 1

vw.release()
cap.release()
print('wrote %s  (%d frames, vehicle colour corrected in %d)' % (OUT, n, fixed))
if shift_before:
    print('  vehicle R-B: %+.1f -> %+.1f   (CARLA is the reference)'
          % (float(np.mean(shift_before)), float(np.mean(shift_after))))

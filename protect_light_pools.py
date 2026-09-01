"""Put CARLA's own night lighting back into the render.

The night model was trained on real dashcam footage, where streetlights are everywhere. CARLA's
Town10HD at night has far fewer light sources, so wherever CARLA gives no lighting cue the model
falls back on "night = black" and fills the area with a void. The result reads as patches of
black next to lit pools rather than a continuous dark scene.

CARLA knows exactly where its light is. This takes the lit regions straight from the night
capture -- lamp pools on the road, headlight cones, glow on nearby walls -- and blends them into
the render, so the illuminated areas are illuminated for the right reason and in the right place.

Design notes:
  - Light is taken as the capture's luminance ABOVE a local background, not its absolute value.
    A white wall in shadow is bright-ish but not a light source; a lamp pool is bright relative
    to the road two metres away. The local background is a heavy blur.
  - Blending is additive in luminance only, with the render's own colour preserved, so the model
    keeps authorship of surfaces and CARLA only supplies where the photons landed.
  - Sky is excluded via the label map. CARLA's night sky is near-black and adding it would
    darken the render's sky, which a separate grade already handles.
  - Highlights are compressed rather than clipped, so lamp cores stay round instead of becoming
    flat white discs.

CARLA PNGs are BGR-as-RGB, so the channels are swapped on read (see carla_rgb_channel_swap).

Usage: protect_light_pools.py in.mp4 carla_rgb_dir label_dir out.mp4 [gain=0.8] [thresh=18]
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from vidcodec import fourcc_for   # .avi -> lossless FFV1, anything else -> mp4v
import cv2, glob, sys, os, numpy as np
from PIL import Image

IN, RGB_DIR, LBL_DIR, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
GAIN = float(sys.argv[5]) if len(sys.argv) > 5 else 0.8
THRESH = float(sys.argv[6]) if len(sys.argv) > 6 else 18.0
SKY = 27

rgbs = sorted(glob.glob(RGB_DIR + '/*.png'))
labs = sorted(glob.glob(LBL_DIR + '/*.png'))
cap = cv2.VideoCapture(IN)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
W, H = int(cap.get(3)), int(cap.get(4))
w = cv2.VideoWriter(OUT, fourcc_for(OUT), fps, (W, H))

n = 0
applied = 0
lit_frac = []
while True:
    ok, im = cap.read()
    if not ok:
        break
    if n < len(rgbs):
        car = cv2.imread(rgbs[n])
        if car is not None:
            car = np.ascontiguousarray(car[:, :, ::-1])          # BGR-as-RGB -> real BGR
            car = cv2.resize(car, (W, H), interpolation=cv2.INTER_AREA)
            cl = cv2.cvtColor(car, cv2.COLOR_BGR2GRAY).astype(np.float32)
            # light = how much brighter than the local surroundings
            bg = cv2.GaussianBlur(cl, (0, 0), 45.0)
            light = np.clip(cl - bg - THRESH, 0, None)
            if labs and n < len(labs):
                lab = np.array(Image.open(labs[n]))
                lab = lab if lab.ndim == 2 else lab[:, :, 0]
                lab = cv2.resize(lab, (W, H), interpolation=cv2.INTER_NEAREST)
                light[lab == SKY] = 0
            if light.max() > 1:
                light = cv2.GaussianBlur(light, (0, 0), 3.0)     # soften the falloff
                lit_frac.append(float((light > 2).mean()) * 100)
                f = im.astype(np.float32)
                y = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
                add = light * GAIN
                # Compress near saturation only. The previous curve,
                #     255*(1-exp(-(y+add)/160)) / (1-exp(-255/160))
                # was NOT the identity at add=0: a pixel at 40 came out at 70.8, so the whole
                # frame was brightened ~1.8x before any light was composited, and the gain
                # parameter barely mattered. This version leaves anything below the knee exactly
                # as it was and rolls off smoothly above it, so only pixels that actually
                # received light change.
                KNEE = 200.0
                t = y + add
                newy = np.where(t < KNEE, t, KNEE + (255.0 - KNEE) * (1.0 - np.exp(-(t - KNEE) / (255.0 - KNEE))))
                scale = np.where(y > 2, newy / np.maximum(y, 1e-3), 1.0 + add / 255.0)
                im = np.clip(f * scale[:, :, None], 0, 255).astype(np.uint8)
                applied += 1
    w.write(im)
    n += 1

w.release()
cap.release()
print('wrote %s  (%d frames, light composited in %d, mean lit area %.1f%%)'
      % (OUT, n, applied, np.mean(lit_frac) if lit_frac else 0.0))

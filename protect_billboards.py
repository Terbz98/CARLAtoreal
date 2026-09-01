"""Restore billboard ARTWORK (e.g. the Coca-Cola logo) from the CARLA render.

The generator only ever sees label + edge + depth + normal. Labelling a region `billboard`
tells it "a sign is here" but nothing about what is printed on it, so it paints a plausible
generic panel. The actual artwork exists only in CARLA's RGB pixels, which the model never
sees. No label taxonomy can encode "red disc with white script".

So take it from CARLA directly -- the same move protect_traffic_lights.py makes for lights.
The difference: a traffic light is copied from the render itself (same domain), whereas CARLA
RGB is flat, clean and game-like. Pasting it raw reads as a sticker. Instead:

  * keep CARLA's CHROMA (a,b in LAB) -- that is the logo's colour, the whole point
  * keep CARLA's DETAIL (L minus its local mean) -- the lettering and shapes
  * take the RENDER's local brightness and contrast (L mean/std) -- so the panel sits under
    the same lighting as the wall it is mounted on
  * feather the mask edge so it does not cut a hard rectangle out of the facade

ALPHA blends the result against the render, so the model's own surface texture still shows
through: 1.0 = fully CARLA artwork, 0.0 = untouched.

Usage:
  python protect_billboards.py <in_mp4> <carla_rgb_dir> <rich_label_dir> <out_mp4> [alpha=0.85] [feather=3]
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from vidcodec import fourcc_for   # .avi -> lossless FFV1, anything else -> mp4v
import sys, os, glob, cv2, numpy as np
from PIL import Image

IN, RGBDIR, LBL, OUT = sys.argv[1:5]
ALPHA = float(sys.argv[5]) if len(sys.argv) > 5 else 0.85
FEATHER = int(sys.argv[6]) if len(sys.argv) > 6 else 3
BILLBOARD = 35                     # Mapillary 'billboard'
MIN_PX = 400                       # ignore specks; a sign worth restoring is bigger
CHROMA = float(os.environ.get('BB_CHROMA', '0.68'))   # 1.0 = raw CARLA colour, 0 = greyscale

labels = sorted(glob.glob(os.path.join(LBL, '*.png')))
rgbs = sorted(glob.glob(os.path.join(RGBDIR, '*.png'))) or sorted(glob.glob(os.path.join(RGBDIR, '*.jpg')))
cap = cv2.VideoCapture(IN)
W = int(cap.get(3)); H = int(cap.get(4)); fps = cap.get(5) or 30
vw = cv2.VideoWriter(OUT, fourcc_for(OUT), fps, (W, H))
print('frames %d  labels %d  rgb %d  alpha %.2f' % (int(cap.get(7)), len(labels), len(rgbs), ALPHA))

i = hit = 0
while True:
    ok, fr = cap.read()
    if not ok:
        break
    if i < len(labels) and i < len(rgbs):
        L = np.array(Image.open(labels[i])); L = L if L.ndim == 2 else L[:, :, 0]
        m = (L == BILLBOARD)
        if m.sum() >= MIN_PX:
            m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
            car = cv2.imread(rgbs[i])
            if car is not None:
                # CARLA's PNG is BGR-as-RGB, so cv2 reads R and B transposed (verified on
                # label-masked sky: B 166 G 186 R 205 -- the sky reads red). Without this the
                # Coca-Cola logo composites BLUE. It did, in the first version of this script.
                car = np.ascontiguousarray(car[:, :, ::-1])
                if car.shape[:2] != (H, W):
                    car = cv2.resize(car, (W, H), interpolation=cv2.INTER_AREA)
                mb = m.astype(bool)
                clab = cv2.cvtColor(car, cv2.COLOR_BGR2LAB).astype(np.float32)
                rlab = cv2.cvtColor(fr, cv2.COLOR_BGR2LAB).astype(np.float32)
                cL = clab[:, :, 0][mb]; rL = rlab[:, :, 0][mb]
                # CARLA's structure, the render's exposure
                shifted = (cL - cL.mean()) / (cL.std() + 1e-6) * (rL.std() + 1e-6) + rL.mean()
                out = clab.copy()
                out[:, :, 0][mb] = shifted            # detail from CARLA, level from render
                # a,b stay CARLA's -> the logo keeps its colour
                # CARLA's texture is a clean game asset: measured 1.7x the saturation of the
                # surrounding render (124.5 vs 73.2), so an untouched paste pops off the wall
                # like a sticker. Pull the chroma toward neutral so the sign sits in the same
                # colour world as the scene. It stays the MOST saturated thing in frame -- a
                # billboard should be -- just not twice over. Structure is in L and untouched,
                # so legibility is unaffected.
                if CHROMA != 1.0:
                    out[:, :, 1][mb] = 128 + (out[:, :, 1][mb] - 128) * CHROMA
                    out[:, :, 2][mb] = 128 + (out[:, :, 2][mb] - 128) * CHROMA
                car_fit = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
                w = cv2.GaussianBlur(m.astype(np.float32), (0, 0), FEATHER)[..., None] * ALPHA
                fr = (fr.astype(np.float32) * (1 - w) + car_fit.astype(np.float32) * w)
                fr = np.clip(fr, 0, 255).astype(np.uint8)
                hit += 1
    vw.write(fr); i += 1
vw.release()
print('wrote %s  (%d frames, billboards restored in %d)' % (OUT, i, hit))

"""Composite lane markings from CARLA, with guaranteed contrast against the road.

WHY: measured on Town10HD finals, lane-marking brightness RELATIVE TO THE SURROUNDING ROAD --
the only thing a lane detector actually keys on:
      v43   +4.4   (white paint, but barely above asphalt)
      v44   -3.2   (markings render DARKER than the road: inverted)
Real white paint on grey asphalt is roughly +40 to +80. Neither model has ever drawn lane
markings properly; v44's wide edge channel merely tipped it negative.

The generator cannot fix this from labels. It knows "lane marking" is here but has no notion
that paint must out-luminance asphalt, and its training photos are full of worn, shadowed,
wet and repainted markings averaging far less contrast than a clean CARLA line.

So take the geometry from CARLA and ENFORCE the contrast, the same trade already made for
traffic lights and billboards: for a lane/LKAS consumer, accurate beats photoreal-but-wrong.

  * shape and position come from CARLA's render (pixel-exact, from the sim's own camera)
  * luminance is rescaled so the marking sits BOOST above the local road level, measured
    per frame from the actual road pixels next to it -- so it tracks scene lighting and
    shadows instead of being a fixed grey
  * chroma is pulled toward neutral (markings are white/yellow, never saturated)
  * a 1px feather keeps the edge from aliasing without softening the line

Usage:
  python protect_lane_markings.py <in_mp4> <carla_rgb_dir> <label_dir> <out_mp4>
                                  [boost=45] [alpha=0.9]
Env: LM_CLASSES (default 8,23,24), LM_FEATHER
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from vidcodec import fourcc_for   # .avi -> lossless FFV1, anything else -> mp4v
import sys, os, glob, cv2, numpy as np
from scipy import ndimage
from PIL import Image

IN, RGBDIR, LBL, OUT = sys.argv[1:5]
BOOST = float(sys.argv[5]) if len(sys.argv) > 5 else 45.0
ALPHA = float(sys.argv[6]) if len(sys.argv) > 6 else 0.9
CLASSES = [int(x) for x in os.environ.get('LM_CLASSES', '8,23,24').split(',')]
ROAD = 13
FEATHER = float(os.environ.get('LM_FEATHER', '1.0'))
MIN_PX = 300
CHROMA = 0.35          # markings are near-neutral; kill CARLA's colour cast

labels = sorted(glob.glob(os.path.join(LBL, '*.png')))
rgbs = (sorted(glob.glob(os.path.join(RGBDIR, '*.png')))
        or sorted(glob.glob(os.path.join(RGBDIR, '*.jpg'))))
cap = cv2.VideoCapture(IN)
W = int(cap.get(3)); H = int(cap.get(4)); fps = cap.get(5) or 30
vw = cv2.VideoWriter(OUT, fourcc_for(OUT), fps, (W, H))
print('frames %d  labels %d  rgb %d  classes %s  boost +%.0f'
      % (int(cap.get(7)), len(labels), len(rgbs), CLASSES, BOOST))

i = hit = 0; before = []; after = []
while True:
    ok, fr = cap.read()
    if not ok:
        break
    if i < len(labels) and i < len(rgbs):
        L = np.array(Image.open(labels[i])); L = L if L.ndim == 2 else L[:, :, 0]
        m0 = np.isin(L, CLASSES)
        if m0.sum() >= MIN_PX:
            m = cv2.resize(m0.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            lroad = cv2.resize((L == ROAD).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            # road immediately beside the marking = the reference level for "how bright is
            # the asphalt right here", so shadows and exposure changes are followed
            near = ndimage.binary_dilation(m, iterations=9) & ~ndimage.binary_dilation(m, iterations=3) & lroad
            car = cv2.imread(rgbs[i])
            if car is not None and near.sum() > 200:
                car = np.ascontiguousarray(car[:, :, ::-1])          # CARLA PNG is BGR-as-RGB
                if car.shape[:2] != (H, W):
                    car = cv2.resize(car, (W, H), interpolation=cv2.INTER_AREA)
                lab = cv2.cvtColor(fr, cv2.COLOR_BGR2LAB).astype(np.float32)
                clab = cv2.cvtColor(car, cv2.COLOR_BGR2LAB).astype(np.float32)
                road_L = lab[:, :, 0][near].mean()
                cL = clab[:, :, 0][m]
                before.append(float(lab[:, :, 0][m].mean() - road_L))
                # keep CARLA's within-marking variation, re-seat it at road + BOOST
                sd = cL.std() + 1e-6
                newL = (cL - cL.mean()) / sd * min(sd, 12.0) + road_L + BOOST
                out = clab.copy()
                out[:, :, 0][m] = np.clip(newL, 0, 255)
                out[:, :, 1][m] = 128 + (out[:, :, 1][m] - 128) * CHROMA
                out[:, :, 2][m] = 128 + (out[:, :, 2][m] - 128) * CHROMA
                fit = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
                w = cv2.GaussianBlur(m.astype(np.float32), (0, 0), FEATHER)[..., None] * ALPHA
                fr = np.clip(fr.astype(np.float32) * (1 - w) + fit.astype(np.float32) * w, 0, 255).astype(np.uint8)
                lab2 = cv2.cvtColor(fr, cv2.COLOR_BGR2LAB).astype(np.float32)
                after.append(float(lab2[:, :, 0][m].mean() - lab2[:, :, 0][near].mean()))
                hit += 1
    vw.write(fr); i += 1
vw.release()
print('wrote %s  (%d frames, markings composited in %d)' % (OUT, i, hit))
if before and after:
    print('lane marking vs road:  %+.1f  ->  %+.1f   (target +%.0f, real paint is +40..+80)'
          % (np.mean(before), np.mean(after), BOOST))

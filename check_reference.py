#!/usr/bin/env python3
"""
Verify a clip and a CARLA recording are the SAME take before compositing anything from one
into the other.

Why this exists: protect_light_pools / protect_lane_markings / protect_billboards /
protect_traffic_lights all paste content sampled from a recording's RGB into a rendered clip,
indexed by frame number. Point them at a different drive and they will happily paint that
drive's lamp pools, lane paint and signage over this one -- producing outlines of objects that
were never in the scene. It fails silently: every stage reports success. This happened on
2026-08-20 (a repost used recorded_Town10HD_night_vp55, a 13 Aug drive, against a clip rebuilt
from recorded_Town10HD_night_inst of 19 Aug) and it is the same class of fault as the stale
input channels that produced a whole tumbling clip.

The check is appearance-invariant on purpose. A render and its source capture look nothing alike
in colour or exposure, but they share scene GEOMETRY, so gradient structure correlates strongly
for the same take and weakly for a different one.

  usage: check_reference.py <clip.mp4|avi> <recording_rgb_dir> [--min 0.30] [--samples 12]
  exit 0 = same take, exit 1 = mismatch (or unreadable)
"""
import cv2, glob, os, sys
import numpy as np

ap = sys.argv
clip, rgbdir = ap[1], ap[2]
MIN = float(ap[ap.index('--min') + 1]) if '--min' in ap else 0.30
N = int(ap[ap.index('--samples') + 1]) if '--samples' in ap else 12


def struct(img, size=(96, 48)):
    """Downscaled gradient magnitude, zero-mean unit-norm -- geometry, not appearance."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = cv2.resize(g, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    m = np.sqrt(gx * gx + gy * gy)
    m -= m.mean()
    n = np.linalg.norm(m)
    return m / n if n > 1e-6 else m


rgbs = sorted(glob.glob(os.path.join(rgbdir, '*.png'))) or \
       sorted(glob.glob(os.path.join(rgbdir, '*.jpg')))

# the clip side may be a video OR a directory of generator frames, depending on which stage
# of the chain is asking
if os.path.isdir(clip):
    frames = sorted(glob.glob(os.path.join(clip, '*_synthesized_image.jpg'))) or \
             sorted(glob.glob(os.path.join(clip, '*.jpg'))) or \
             sorted(glob.glob(os.path.join(clip, '*.png')))
    cap, total = None, len(frames)
else:
    frames = None
    cap = cv2.VideoCapture(clip)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if not rgbs or total < 2:
    print(f'REFERENCE CHECK: cannot read (clip frames={total}, recording images={len(rgbs)})')
    sys.exit(1)
if abs(total - len(rgbs)) > max(5, 0.02 * len(rgbs)):
    print(f'REFERENCE CHECK: FAIL -- clip has {total} frames, recording has {len(rgbs)}')
    sys.exit(1)

idx = np.linspace(0, min(total, len(rgbs)) - 1, N).astype(int)
scores = []
for i in idx:
    if frames is not None:
        fr = cv2.imread(frames[int(i)]); ok = fr is not None
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
    ref = cv2.imread(rgbs[int(i)])
    if not ok or ref is None:
        continue
    # CARLA writes BGRA and the PNGs read back BGR-as-RGB; luminance is channel-order
    # invariant and this only uses gradients, so no swap is needed
    scores.append(float((struct(fr) * struct(ref)).sum()))
if cap is not None:
    cap.release()

if not scores:
    print('REFERENCE CHECK: FAIL -- no comparable frames')
    sys.exit(1)
med = float(np.median(scores))
print(f'REFERENCE CHECK: median structural correlation {med:.3f} over {len(scores)} frames '
      f'(threshold {MIN:.2f})  [{os.path.basename(clip)} vs {os.path.basename(rgbdir.rstrip("/"))}]')
if med < MIN:
    print('REFERENCE CHECK: FAIL -- clip and recording are not the same take')
    sys.exit(1)
print('REFERENCE CHECK: PASS')

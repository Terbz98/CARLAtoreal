#!/usr/bin/env python3
"""Scale one source's saturation to hit a TARGET, instead of applying a fixed multiplier to it.

WHY NOT A FIXED MULTIPLIER. fix_zod_appearance.py takes --sat, an amount to multiply by, which is
right when you are correcting one known source. It is wrong when you are normalising several
sources to a common look, because they do not start in the same place. Measured on PandaSet's three
forward cameras, daylight frames only, the same x1.60 lands in three different places:

    camera                base    x1.60    wanted
    front_camera          37.0     59.9      61.6
    front_left_camera     46.4     75.0      61.6
    front_right_camera    45.3     73.1      61.6

The two side cameras overshoot Mapillary's 61.6 by more than ten points, and oversaturation is a
defect the user has already reported once: v75q's cars ran at 2.10x CARLA's saturation and they
named it on sight. The error came from calibrating the multiplier against the UNGATED corpus (base
37.8, all three cameras and the night frames together) and then applying it after the daylight gate,
where the base is 45.3.

So measure each source where it actually sits, and solve for the factor. One target, one look.

Saturation only: no unsharp. PandaSet's detail is already the highest in the corpus and sharpening
it further would only invite the ringing that the clamped unsharp in fix_zod_appearance.py exists
to avoid.

  usage: match_saturation.py <src_dir> <dst_dir> --target 61.6 [--prefix pshr_] [--max 2.0]
"""
import os
import sys

import cv2
import numpy as np

SRC, DST = sys.argv[1], sys.argv[2]


def arg(name, default):
    return float(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


TARGET = arg('--target', 61.6)
MAXF = arg('--max', 2.0)
PREFIX = sys.argv[sys.argv.index('--prefix') + 1] if '--prefix' in sys.argv else ''

files = sorted(f for f in os.listdir(SRC)
               if f.startswith(PREFIX) and f.lower().endswith(('.jpg', '.jpeg', '.png')))
if not files:
    raise SystemExit(f'no images matching {PREFIX!r} in {SRC}')
os.makedirs(DST, exist_ok=True)

# measure on a sample, apply to all
sample = files[::max(1, len(files) // 300)][:300]
base = float(np.mean([cv2.cvtColor(cv2.imread(os.path.join(SRC, f)), cv2.COLOR_BGR2HSV)[..., 1].mean()
                      for f in sample]))
factor = min(MAXF, TARGET / max(base, 1e-3))
print(f'  {len(files)} frames, measured sat {base:.1f}, target {TARGET:.1f}, factor x{factor:.3f}')

n = 0
for f in files:
    a = cv2.imread(os.path.join(SRC, f))
    if a is None:
        continue
    hsv = cv2.cvtColor(a, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    cv2.imwrite(os.path.join(DST, f), out,
                [cv2.IMWRITE_JPEG_QUALITY, 97] if f.lower().endswith(('.jpg', '.jpeg')) else [])
    n += 1

# Verify on the SAME sample the base was measured on. Measuring "after" on a different subset (an
# every-200th-file stride over a 300-file sample, which is two frames) reported 86.6 against a
# target of 61.6 and looked like a broken transform rather than a broken measurement.
after = float(np.mean([cv2.cvtColor(cv2.imread(os.path.join(DST, f)), cv2.COLOR_BGR2HSV)[..., 1].mean()
                       for f in sample]))
# clipping at 255 means the achieved value undershoots the arithmetic one on saturated sources
print(f'  wrote {n} to {DST}, sat {base:.1f} -> {after:.1f} (target {TARGET:.1f})')
if n != len(files):
    raise SystemExit(f'wrote {n} of {len(files)} -- refusing to report success')

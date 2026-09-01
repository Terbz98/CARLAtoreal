#!/usr/bin/env python3
"""Reject a CARLA take where the ego was respawned mid-clip, by finding frames with no view.

record_town_auto.py respawns the ego when it has been boxed in for 5 seconds. What that actually
looks like — confirmed by eye on both affected takes — is not a clean jump to another street. The
camera ends up buried in geometry: on Town10HD the whole frame becomes close-up concrete from
frame 709, and on Town03 it is jammed against the back of an SUV at 208-209. The drive then
resumes elsewhere, with a DIFFERENT ego vehicle. Observed on Town10HD at 0:22.

TWO DETECTORS THAT DID NOT WORK, recorded so they are not retried:
  1. The speed trace. Town10HD's largest speed change (1.67) is SMALLER than clean Town04's, and
     the "stopped then instantly moving" pattern appears in no take at all.
  2. Frame-to-frame image difference. The cut is a real spike (63.8 against neighbours at 9-12)
     but Town10HD is a fast drive whose ordinary frame differences already run 15-25, so no
     single threshold separates it from a fast turn without also missing the cut. It caught
     Town03 and missed Town10HD.

WHAT IS EXACT: the SEMANTIC map. A camera buried in a surface sees exactly one class — frame 709's
semantic map has a single unique value across the entire image — whereas any real driving view
contains many (road, building, sky, vehicles...). This needs no threshold tuning at all: it is a
count of distinct classes, and a driving frame never has one.

CARLA writes the tag in the BLUE channel of the semantic PNG.

  usage: check_teleport.py <recording_dir> [min_classes=4]
  exit 0 = clean, exit 1 = obscured frames found
"""
import glob
import os
import sys

import cv2
import numpy as np

REC = sys.argv[1]
MIN_CLASSES = int(sys.argv[2]) if len(sys.argv) > 2 else 4

sem = sorted(glob.glob(os.path.join(REC, 'semantic', '*.png')))
if len(sem) < 200:
    print(f'  check_teleport: only {len(sem)} semantic frames in {REC}')
    sys.exit(1)

bad = []
for i, f in enumerate(sem):
    im = cv2.imread(f, cv2.IMREAD_UNCHANGED)
    if im is None:
        continue
    tag = im[:, :, 0] if im.ndim == 3 else im       # blue channel holds the CARLA tag
    small = tag[::4, ::4]
    vals, cnt = np.unique(small, return_counts=True)
    # obscured = almost no class variety, or one class swallowing essentially the whole frame
    if len(vals) < MIN_CLASSES or (cnt.max() / cnt.sum()) > 0.985:
        bad.append(i)

events = []
for b in bad:
    if not events or b - events[-1][-1] > 5:
        events.append([b])
    else:
        events[-1].append(b)

if events:
    desc = ', '.join(f'{e[0]}-{e[-1]}' for e in events[:5])
    print(f'  check_teleport: {len(sem)} frames, {len(bad)} obscured in {len(events)} event(s): {desc}')
    sys.exit(1)
print(f'  check_teleport: {len(sem)} frames, no obscured frames — clean')
sys.exit(0)

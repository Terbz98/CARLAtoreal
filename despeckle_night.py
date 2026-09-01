#!/usr/bin/env python3
"""Remove the faint white specks from night clips without touching real light sources.

Reported defect: "very faint shiny white dots" on night. Measured on the delivered v51 clips:
3,008 specks/frame on town03 and 3,671 on town10hd, against 1,801 for sunny v50 — night is the
worse case, and it never received any of the cleanup sunny got.

A speck must be distinguished from a lamp, a tail light or a lit window, because destroying those
would be far worse than the dots. Three conditions together, all of which a real light fails:

  BRIGHTER THAN ITS SURROUNDINGS   more than `thresh` grey levels above the 5x5 local median.
                                   A large lamp is bright in absolute terms but its centre is
                                   close to its own neighbourhood, so it does not qualify.
  TINY                             connected area under `max_area` px. Streetlights, headlights,
                                   traffic lamps and windows are all far larger.
  NEARLY COLOURLESS                low saturation. Tail lights, traffic signals and sodium lamps
                                   are strongly coloured; the reported artefact is white.

Qualifying pixels are replaced with their local median, i.e. the surrounding surface, so nothing
is blurred and no edge moves.

  usage: despeckle_night.py <in> <out> [thresh=34] [max_area=14] [max_sat=70]
"""
import sys, os
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vidcodec import fourcc_for

IN, OUT = sys.argv[1], sys.argv[2]
THRESH = int(sys.argv[3]) if len(sys.argv) > 3 else 34
MAX_AREA = int(sys.argv[4]) if len(sys.argv) > 4 else 14
MAX_SAT = int(sys.argv[5]) if len(sys.argv) > 5 else 70

cap = cv2.VideoCapture(IN)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
w = cv2.VideoWriter(OUT, fourcc_for(OUT), fps, (W, H))

n = 0
removed_before = removed_after = 0
frac = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    med = cv2.medianBlur(g, 5)
    excess = g.astype(np.int16) - med.astype(np.int16)
    sat = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)[:, :, 1]
    cand = ((excess > THRESH) & (sat < MAX_SAT)).astype(np.uint8)

    removed_before += int((excess > 40).sum())
    if cand.any():
        nl, lab, st, _ = cv2.connectedComponentsWithStats(cand, 8)
        keep = np.zeros(nl, bool)
        for i in range(1, nl):
            keep[i] = st[i, cv2.CC_STAT_AREA] <= MAX_AREA     # tiny only; lamps survive
        mask = keep[lab]
        if mask.any():
            out = f.copy()
            m3 = cv2.medianBlur(f, 5)
            out[mask] = m3[mask]
            f = out
            frac.append(float(mask.mean()))
    g2 = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    removed_after += int(((g2.astype(np.int16) - cv2.medianBlur(g2, 5).astype(np.int16)) > 40).sum())
    w.write(f)
    n += 1
w.release(); cap.release()
pf = (np.mean(frac) * 100) if frac else 0.0
print(f'  despeckle: {n} frames, {pf:.3f}% of pixels replaced, '
      f'specks {removed_before/max(n,1):.0f} -> {removed_after/max(n,1):.0f} per frame')

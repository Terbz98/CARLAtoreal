#!/usr/bin/env python3
"""Give buildings their real structure back, from CARLA, instead of smoothing the invented one.

WHY THIS AND NOT MORE FILTERING. Buildings are 25% of the frame and own more flicker than every
other class combined (mean 39.4 against road's 3.1 in the delivered baseline). Every filtering
attempt hit the same wall: a temporal median cannot tell "wrong detail" from "detail", so removing
the shimmer removes the windows with it. Measured on Town10HD, a 5-frame class-weighted median cut
building flicker 70% and visual review returned "very very blurry, it lost all the details" -- and
they were right: the metric fell because a smear is temporally consistent, not because the picture
improved.

The root cause is not the filter. The label map says "building" and nothing more -- no window
layout, no facade pattern -- so the generator invents the detail, and invents it differently every
frame. That difference IS the flicker. No amount of post-processing can fix it, because the
information was never there.

CARLA has it. The simulator renders real window grids, perfectly stable frame to frame, from the
same camera as the label map, so it is already aligned. This is the same move that fixed traffic
lights, lane markings, billboards and vehicle colour: take from the simulator what the model
cannot know, and keep from the model what it is good at.

METHOD -- replace the high frequencies, keep everything else:
    out_L = blur(render_L) + STRENGTH * highpass(carla_L)
The render keeps its low frequencies (lighting, exposure, weathering, colour grade) and its own
a/b chroma, so the building still looks photoreal rather than like a game asset. Only the fine
structure is swapped, and that structure is now identical in every frame, so the flicker it used
to carry is gone -- without a temporal filter, and without losing detail. Detail should go UP.

  usage: protect_buildings.py <in> <carla_rgb_dir> <label_dir> <out> [strength=0.8] [sigma=2.0]
"""
import glob
import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vidcodec import fourcc_for

IN, RGBDIR, LABDIR, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
STRENGTH = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
SIGMA = float(sys.argv[6]) if len(sys.argv) > 6 else 1.4

# Mapillary-65: 17 building, 6 wall, 3 barrier-ish. Only large static built surfaces -- never
# vegetation (CARLA's foliage is a poor match for the render's) and never road or vehicles.
BUILDING_IDS = [17, 6]
FEATHER = 3.0
MIN_PX = 4000          # ignore slivers; a facade worth restoring is large

rgbs = sorted(glob.glob(os.path.join(RGBDIR, '*.png')))
labs = sorted(glob.glob(os.path.join(LABDIR, '*.png')))
cap = cv2.VideoCapture(IN)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
vw = cv2.VideoWriter(OUT, fourcc_for(OUT), fps, (W, H))

n = hit = skipped = 0
det_before, det_after = [], []
while True:
    ok, fr = cap.read()
    if not ok:
        break
    if n < len(rgbs) and n < len(labs):
        car = cv2.imread(rgbs[n])
        lab = np.array(Image.open(labs[n]))
        if car is not None:
            # CARLA writes BGRA and the slice leaves BGR-as-RGB; reverse to real BGR
            car = np.ascontiguousarray(car[:, :, ::-1])
            car = cv2.resize(car, (W, H), interpolation=cv2.INTER_AREA)
            lab = lab if lab.ndim == 2 else lab[:, :, 0]
            lab = cv2.resize(lab, (W, H), interpolation=cv2.INTER_NEAREST)
            m = np.isin(lab, BUILDING_IDS).astype(np.uint8)
            # pull in from the silhouette so sky/foreground never bleeds into the transfer
            m = cv2.erode(m, np.ones((5, 5), np.uint8), iterations=1)
            if m.sum() > MIN_PX:
                mb = m.astype(bool)
                rlab = cv2.cvtColor(fr, cv2.COLOR_BGR2LAB).astype(np.float32)
                clab = cv2.cvtColor(car, cv2.COLOR_BGR2LAB).astype(np.float32)
                rL, cL = rlab[:, :, 0], clab[:, :, 0]
                lap_before = cv2.Laplacian(rL, cv2.CV_32F)   # scored later on the SAME pixels
                                                            # that end up being modified

                base = cv2.GaussianBlur(rL, (0, 0), SIGMA)          # render's lighting and grade
                detail = cL - cv2.GaussianBlur(cL, (0, 0), SIGMA)   # CARLA's real window structure
                # NO amplitude matching. The first version scaled CARLA's detail to the render's
                # level, and CARLA measurably has MORE facade structure (Laplacian variance 2478
                # vs 1434, high-pass energy 472 vs 259) -- so that scaled the good detail DOWN
                # while the blur above removed the render's, and the result lost 38% of its
                # detail. The whole point is that CARLA knows what the building looks like and
                # the generator does not, so inject it at full strength.
                # PER-REGION, OUTCOME-BASED GUARD.
                # A whole-frame decision was not enough. With it, facade detail still fell on two
                # towns (town05 -12%, town06 -11%) while rising on three (town10hd +20%, town03
                # +14%, town04 +8%): a frame containing one richly modelled facade and three plain
                # ones passes the frame-level test and then loses detail on the plain three.
                #
                # So decide per connected building, and decide on the OUTCOME rather than on a
                # proxy: build the merged version, measure its detail against the original for
                # that region, and keep whichever is actually sharper. A proxy can be wrong; the
                # result cannot. This stage can then never reduce detail anywhere -- at worst it
                # does nothing.
                merged_full = base + detail * STRENGTH
                out = rlab.copy()
                nl, cc, st_, _ = cv2.connectedComponentsWithStats(m, 8)
                used = np.zeros_like(m, dtype=bool)
                for ci in range(1, nl):
                    if st_[ci, cv2.CC_STAT_AREA] < MIN_PX:
                        continue
                    reg = cc == ci
                    before = float(cv2.Laplacian(rL, cv2.CV_32F)[reg].var())
                    after = float(cv2.Laplacian(merged_full, cv2.CV_32F)[reg].var())
                    if after > before:
                        out[:, :, 0][reg] = np.clip(merged_full[reg], 0, 255)
                        used |= reg
                    else:
                        skipped += 1
                if not used.any():
                    vw.write(fr); n += 1
                    continue
                mb = used
                # Score before and after on the SAME pixel set. The first version accumulated
                # `before` over the whole building mask and `after` over only the regions actually
                # modified, so a frame where the guard declined most regions reported a fall that
                # never happened -- town06 printed -14% when the delivered clip measures 0.97x on
                # identical pixels.
                det_before.append(float(lap_before[used].var()))

                # a,b untouched throughout: the colour stays the render's
                fit = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
                wgt = cv2.GaussianBlur(used.astype(np.float32), (0, 0), FEATHER)[..., None]
                fr = np.clip(fr.astype(np.float32) * (1 - wgt) + fit.astype(np.float32) * wgt,
                             0, 255).astype(np.uint8)
                det_after.append(float(cv2.Laplacian(
                    cv2.cvtColor(fr, cv2.COLOR_BGR2LAB).astype(np.float32)[:, :, 0],
                    cv2.CV_32F)[mb].var()))
                hit += 1
    vw.write(fr)
    n += 1

vw.release(); cap.release()
b = np.mean(det_before) if det_before else 0
a = np.mean(det_after) if det_after else 0
print(f'  buildings: {n} frames, structure restored in {hit}, '
      f'{skipped} skipped where CARLA had less detail; '
      f'facade detail {b:.0f} -> {a:.0f} ({(a/b-1)*100:+.0f}%)')

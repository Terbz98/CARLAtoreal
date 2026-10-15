#!/usr/bin/env python3
"""Put CARLA's ground shadows back into a delivered clip, deeper than CARLA renders them.

THE DEFECT. The user has said three times that v50m's shadows are the best in the project and that
nothing since comes close. Measured on ROAD pixels, Town10HD, as the darkest twentieth of the road
over its own median (lower = deeper shadow):

    CARLA source   0.542        v50m   0.269        v75q   0.558
    v85 raw        0.659        v85q   0.630

v50m's road shadows go twice as deep as CARLA's. Every licensed model sits at 0.56-0.66 -- barely
shadowed at all. And v85's raw render (0.659) and delivered clip (0.630) are almost the same, so the
delivery chain is not creating this and cannot be tuned into creating it.

WHY A POST-PASS AND NOT TRAINING. The information is not in the label map: two identical labels at
noon and at four want different shadows, so the generator can only guess, and v80 showed that
handing it a shadow CHANNEL makes it stop learning the surface (-3.6 CIPO). But CARLA knows exactly
where every shadow falls -- it has the geometry and the light. The same argument that makes
protect_lane_markings and protect_buildings work: take from CARLA only what CARLA is authoritative
about, and here that is the SHAPE of the ground shadow.

WHY DEEPER THAN CARLA. Because CARLA is not the target. v50m matches CARLA's contact ratio worst of
every model tested (0.803 against 0.957) and is the one the user prefers. A rendered hard shadow in
CARLA is filled in by ambient; a real one is not. DEEPEN scales (1 - ratio), so 2.0 turns CARLA's
0.542 into 0.084 below its reference -- the knob is calibrated against v50m, not against CARLA.

WHAT IS TRANSFERRED. A luminance RATIO only, so the render keeps its own colour, grain and exposure.
Road and sidewalk only, from the label map. The reference is the lit level of the same class in the
same band of rows, so the correction follows the road's perspective ramp instead of encoding
distance from the camera -- a square window here measures depth, not shadow, which cost three
attempts to learn while building gen_ground_shadow.py.

  usage: deepen_road_shadows.py in.mp4 carla_rgb_dir label_dir out.mp4 [deepen=2.0] [strength=0.9]
CARLA PNGs are BGR-as-RGB, so channels are swapped on read.
"""
import os
import sys

import cv2
import numpy as np

IN, CARLA_DIR, LABEL_DIR, OUT = sys.argv[1:5]
DEEPEN = float(sys.argv[5]) if len(sys.argv) > 5 else 2.0
STRENGTH = float(sys.argv[6]) if len(sys.argv) > 6 else 0.9
LIMIT = int(sys.argv[7]) if len(sys.argv) > 7 else 0        # 0 = whole clip; else first N frames

GROUND = [13, 15]          # road, sidewalk -- measured as 97% of the lower third of a frame
BAND_PX = 16               # rows per reference band
BAND_SMOOTH = 5
REF_PCT = 75               # percentile of in-class luminance taken as "lit ground"
FLOOR = 0.30               # never darker than this fraction of the render's own lit level
MED = int(os.environ.get('MED', '15'))                  # anything thinner than this cannot survive into the shadow map
# HALOFIX=1 (2026-10-13): the gain map is 1.0 on every non-ground pixel, and the median + blur below then
# pulls the ground right next to each car back toward 1.0 -- exactly where CARLA's contact shadow is
# deepest -- leaving a bright rim around every vehicle (ring 1-3 px outside cars minus ring 7-14 px:
# +4.7 L before this pass, +10.9 after; the user saw "a white border around the cars"). With HALOFIX the
# non-ground holes are filled from the surrounding ground gain (normalised convolution) before smoothing,
# and the result is applied to ground pixels only, through a 1.5 px feather.
HALOFIX = os.environ.get('HALOFIX', '0') == '1'


def band_reference(g, m):
    """Lit-ground luminance per row: a high percentile of in-class pixels, by band, smoothed."""
    H = g.shape[0]
    nb = max(1, H // BAND_PX)
    ref = np.full(nb, np.nan, np.float32)
    for b in range(nb):
        y0, y1 = b * BAND_PX, min((b + 1) * BAND_PX, H)
        v = g[y0:y1][m[y0:y1]]
        if v.size >= 64:
            ref[b] = np.percentile(v, REF_PCT)
    if np.all(np.isnan(ref)):
        return None
    idx = np.arange(nb)
    good = ~np.isnan(ref)
    ref = np.interp(idx, idx[good], ref[good])
    k = np.ones(BAND_SMOOTH, np.float32) / BAND_SMOOTH
    ref = np.convolve(np.pad(ref, BAND_SMOOTH // 2, mode='edge'), k, mode='valid')[:nb]
    per_row = np.repeat(ref, BAND_PX)
    if per_row.shape[0] < H:                 # H is not a multiple of BAND_PX in general
        per_row = np.pad(per_row, (0, H - per_row.shape[0]), mode='edge')
    return per_row[:H]


cap = cv2.VideoCapture(IN)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
carla = sorted(f for f in os.listdir(CARLA_DIR) if f.lower().endswith(('.png', '.jpg')))
labels = sorted(f for f in os.listdir(LABEL_DIR) if f.lower().endswith('.png'))

n = changed = 0
total = min(len(carla), len(labels))
if LIMIT:
    total = min(total, LIMIT)
while n < total:
    ok, frame = cap.read()
    if not ok:
        break
    g = cv2.imread(os.path.join(CARLA_DIR, carla[n]))
    lab = cv2.imread(os.path.join(LABEL_DIR, labels[n]), cv2.IMREAD_GRAYSCALE)
    if g is None or lab is None:
        out.write(frame); n += 1; continue
    g = cv2.resize(g[:, :, ::-1], (W, H))                    # BGR-as-RGB -> real BGR
    lab = cv2.resize(lab, (W, H), interpolation=cv2.INTER_NEAREST)

    gl = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY).astype(np.float32)
    fl = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gain = np.ones((H, W), np.float32)
    touched = False

    for cid in GROUND:
        m = (lab == cid)
        if m.sum() < 5000:
            continue
        cref = band_reference(gl, m)                          # CARLA's lit level
        rref = band_reference(fl, m)                          # the render's own lit level
        if cref is None or rref is None:
            continue
        ratio = gl / np.maximum(cref[:, None], 1.0)
        # deepen only where CARLA says there IS shadow; never brighten lit ground
        ratio = np.where(ratio < 1.0, 1.0 - (1.0 - ratio) * DEEPEN, 1.0)
        target = np.clip(rref[:, None] * ratio, 1, 255)
        sub = np.clip(target / np.maximum(fl, 1.0), FLOOR, 1.0)
        gain[m] = sub[m]
        touched = True

    if not touched:
        out.write(frame); n += 1; continue

    if HALOFIX:
        gm = np.isin(lab, GROUND).astype(np.float32)
        den = cv2.GaussianBlur(gm, (0, 0), 12.0)
        fill = cv2.GaussianBlur(gain * gm, (0, 0), 12.0) / np.maximum(den, 1e-3)
        gain = np.where(gm > 0, gain, np.where(den > 1e-3, fill, 1.0)).astype(np.float32)
    # the map is smoothed, not the frame: a median kills anything thinner than a real shadow, and
    # the blur keeps the render's own grain intact because only the GAIN is blurred
    gain = cv2.medianBlur((np.clip(gain, FLOOR, 1.0) * 255).astype(np.uint8), MED).astype(np.float32) / 255.0
    gain = cv2.GaussianBlur(gain, (0, 0), 4.0)
    gain = 1.0 + (gain - 1.0) * STRENGTH
    if HALOFIX:
        feather = cv2.GaussianBlur(gm, (0, 0), 1.5)
        gain = 1.0 + (gain - 1.0) * feather
    res = np.clip(frame.astype(np.float32) * gain[:, :, None], 0, 255)
    if HALOFIX:
        # the render itself draws a bright rim of road around vehicles (+7 L before this pass; v50m +2.5):
        # ground within RIM px of a vehicle may be no brighter than the ground just beyond that band
        VEH = np.isin(lab, [52, 54, 55, 56, 57, 58, 59, 60, 61, 62]).astype(np.uint8)
        if VEH.sum() > 500:
            RIM = int(os.environ.get('RIM', '6'))
            near = (cv2.dilate(VEH, np.ones((2 * RIM + 1, 2 * RIM + 1), np.uint8)) > 0) & (VEH == 0) & (gm > 0)
            ref_m = (gm > 0) & ~near & (VEH == 0)
            Lr = cv2.cvtColor(res.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            den = cv2.GaussianBlur(ref_m.astype(np.float32), (0, 0), 10.0)
            loc = cv2.GaussianBlur(Lr * ref_m, (0, 0), 10.0) / np.maximum(den, 1e-3)
            g2 = np.where(near & (den > 1e-2) & (Lr > loc), loc / np.maximum(Lr, 1.0), 1.0).astype(np.float32)
            g2 = cv2.GaussianBlur(g2, (0, 0), 1.0)
            res = res * g2[:, :, None]
    res = np.clip(res, 0, 255).astype(np.uint8)
    out.write(res)
    changed += 1
    n += 1
    if n % 200 == 0:
        print(f'  {n}/{total}', flush=True)

out.release(); cap.release()
print(f'wrote {OUT}  ({n} frames, road shadows deepened in {changed})')

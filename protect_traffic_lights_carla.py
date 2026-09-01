"""Composite traffic lights straight from CARLA. Accuracy over photorealism.

WHY THIS REPLACES protect_traffic_lights.py
  That script does `fb[m] = fr[m]` -- it copies the light from the RAW MODEL RENDER (pre-DVP)
  purely to undo DVP's temporal washout. It restores the model's OWN traffic light, so every
  defect in it survives:
    * blurry            v42 measured 179-228 Laplacian in TL regions vs CARLA's 784-947 (4-5x)
    * red AND green lit at once
    * wrong housing shape
  The model cannot fix any of these. Its inputs are label + edge + depth + normal, and the
  label says only "traffic light" -- nothing encodes WHICH lamp is lit. Signal state exists
  only in CARLA's RGB pixels. So take it from there, exactly as protect_billboards.py does
  for signage artwork.

DESIGN -- the lamp and the housing are treated differently, because they fail differently:
  * LAMP (saturated + bright in CARLA) is pasted VERBATIM. Its colour is the safety-critical
    information; relighting it would be the very corruption we are removing. A soft additive
    halo is added around it so an emissive source does not read as a flat sticker.
  * HOUSING (the dark metal box) gets its LAB lightness matched to the surrounding render, so
    the fixture sits under scene lighting instead of looking pasted on. Its geometry still
    comes from CARLA, which is the point -- the shape becomes correct.

  Set --verbatim to skip housing relighting entirely (pure CARLA pixels, maximum accuracy,
  slightly more game-like).

DILATION is small (default 2). The old script used 6 to cover render offset plus glow, but the
GT label comes from the same CARLA camera as the RGB, so there is no offset -- and a large
dilation pastes CARLA sky and building around every light, which reads far worse than the
blur it fixes.

Run this LAST, after DVP / v33 / photoreal_post, so nothing downstream can wash it out again.

Usage:
  python protect_traffic_lights_carla.py <in_mp4> <carla_rgb_dir> <label_dir> <out_mp4>
                                         [dilate=2] [alpha=1.0] [--verbatim]
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from vidcodec import fourcc_for   # .avi -> lossless FFV1, anything else -> mp4v
import sys, os, glob, cv2, numpy as np
from scipy import ndimage
from PIL import Image

args = [a for a in sys.argv[1:] if not a.startswith('--')]
VERBATIM = '--verbatim' in sys.argv
IN, RGBDIR, LBL, OUT = args[:4]
DIL = int(args[4]) if len(args) > 4 else 2
ALPHA = float(args[5]) if len(args) > 5 else 1.0
TL = 48                       # Mapillary 'Traffic Light'
MIN_PX = 40                   # a light smaller than this is a few pixels of noise
LAMP_S, LAMP_V = 80, 150      # what counts as a lit lamp in CARLA's clean render
LAMP_FRAC = float(os.environ.get('TL_LAMP_FRAC', '0.16'))  # max lens area as a fraction of its own signal
GLOW = float(os.environ.get('TL_GLOW', '0.22'))       # 0 disables the halo entirely
TL_CLEAN = os.environ.get('TL_CLEAN', '0') == '1'     # redraw lamps as pure red/amber/green discs
FEATHER = 1.0                 # keep tight; a fixture has hard edges

labels = sorted(glob.glob(os.path.join(LBL, '*.png')))
rgbs = (sorted(glob.glob(os.path.join(RGBDIR, '*.png')))
        or sorted(glob.glob(os.path.join(RGBDIR, '*.jpg'))))
cap = cv2.VideoCapture(IN)
W = int(cap.get(3)); H = int(cap.get(4)); fps = cap.get(5) or 30
vw = cv2.VideoWriter(OUT, fourcc_for(OUT), fps, (W, H))
print('frames %d  labels %d  rgb %d  dilate %d  alpha %.2f  %s'
      % (int(cap.get(7)), len(labels), len(rgbs), DIL, ALPHA,
         'verbatim' if VERBATIM else 'relit housing'))

i = hit = lamps = 0
while True:
    ok, fr = cap.read()
    if not ok:
        break
    if i < len(labels) and i < len(rgbs):
        L = np.array(Image.open(labels[i])); L = L if L.ndim == 2 else L[:, :, 0]
        m = (L == TL)
        if m.sum() >= MIN_PX:
            m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            if DIL:
                m = ndimage.binary_dilation(m, iterations=DIL)
            car = cv2.imread(rgbs[i])
            if car is not None:
                # CARLA dumps BGRA and the saved PNG is BGR-as-RGB, so cv2 reads it with R and
                # B transposed -- the same quirk documented in README.md for the semantic camera.
                # Verified on label-masked sky pixels: B 166 G 186 R 205, i.e. the sky reads
                # RED. Without this swap a red light composites as BLUE, which on a traffic
                # signal is the worst possible failure.
                car = np.ascontiguousarray(car[:, :, ::-1])
                if car.shape[:2] != (H, W):
                    car = cv2.resize(car, (W, H), interpolation=cv2.INTER_AREA)
                chsv = cv2.cvtColor(car, cv2.COLOR_BGR2HSV)
                lamp = m & (chsv[:, :, 1] > LAMP_S) & (chsv[:, :, 2] > LAMP_V)
                src = car.astype(np.float32)

                if not VERBATIM:
                    # housing only: CARLA's structure, the render's exposure
                    house = m & ~lamp
                    if house.sum() > 20:
                        clab = cv2.cvtColor(car, cv2.COLOR_BGR2LAB).astype(np.float32)
                        rlab = cv2.cvtColor(fr, cv2.COLOR_BGR2LAB).astype(np.float32)
                        cL = clab[:, :, 0][house]; rL = rlab[:, :, 0][house]
                        clab[:, :, 0][house] = ((cL - cL.mean()) / (cL.std() + 1e-6)
                                                * (rL.std() + 1e-6) + rL.mean())
                        relit = cv2.cvtColor(np.clip(clab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
                        src = np.where(house[..., None], relit.astype(np.float32), src)
                    # the lamp keeps CARLA's exact colour -- that is the signal state

                w = cv2.GaussianBlur(m.astype(np.float32), (0, 0), FEATHER)[..., None] * ALPHA
                fr = fr.astype(np.float32) * (1 - w) + src * w

                # Emissive halo, only around genuinely LIT lamps. The lamp test alone is not
                # enough: CARLA's Town10HD signals have a US-style YELLOW housing, which is
                # itself bright and saturated, so a plain brightness+saturation test tags the
                # whole frame as "lamp" and the halo makes the entire fixture glow gold.
                # A lit lamp is additionally SMALL and COMPACT, so filter on component size.
                # A lamp must be small RELATIVE TO ITS OWN SIGNAL, not smaller than a fixed
                # pixel count: an absolute LAMP_MAX let the yellow housing of a nearby signal
                # (well under 4000 px) qualify, and CLEAN then painted a disc the size of the
                # whole fixture. So work per signal -- find each traffic-light component from
                # the label, and inside it accept only sub-regions that are a small fraction
                # of that housing and roughly round. A US signal shows one lit lens of maybe
                # 3-12% of the fixture area; the housing panel is 30%+ and is rejected.
                if GLOW > 0 and lamp.any():
                    lit = np.zeros_like(lamp)
                    nh, hc, hst, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
                    for hidx in range(1, nh):
                        harea = float(hst[hidx, cv2.CC_STAT_AREA])
                        if harea < 60:
                            continue
                        sub = lamp & (hc == hidx)
                        if not sub.any():
                            continue
                        nl, lc, lst, _ = cv2.connectedComponentsWithStats(sub.astype(np.uint8), 8)
                        cands = []
                        for j in range(1, nl):
                            a = lst[j, cv2.CC_STAT_AREA]
                            bw, bh = lst[j, cv2.CC_STAT_WIDTH], lst[j, cv2.CC_STAT_HEIGHT]
                            if a < 8 or a > harea * LAMP_FRAC:
                                continue
                            if a / float(bw * bh + 1e-6) < 0.45:   # hollow -> a frame, not a lens
                                continue
                            ar = bw / float(bh + 1e-6)
                            if ar < 0.45 or ar > 2.2:              # a lens is round; a panel is not
                                continue
                            cands.append((float(chsv[:, :, 2][lc == j].mean()), j))
                        # at most the two brightest lenses per signal (one lit, sometimes a
                        # second during amber transition) -- never the whole housing
                        for _, j in sorted(cands, reverse=True)[:2]:
                            lit |= (lc == j)
                    if lit.any():
                        # CLEAN mode: redraw each lit lamp as a filled disc in a pure signal
                        # colour. CARLA's lamp is already correct but it is a small, shaded,
                        # partly-specular blob, and after any downscale (the Vision Pilot crop
                        # is 2048 -> 1066 -> 1920) it smears. A detector keys on colour and a
                        # round bright region, so an unambiguous disc is strictly easier to
                        # classify than a faithful one. Position, size and WHICH lamp is lit
                        # all still come from CARLA -- only the rendering is idealised.
                        if TL_CLEAN:
                            hsvc = cv2.cvtColor(car, cv2.COLOR_BGR2HSV)
                            nl2, lc2, lst2, cen2 = cv2.connectedComponentsWithStats(lit.astype(np.uint8), 8)
                            for j in range(1, nl2):
                                cm = (lc2 == j)
                                hs = hsvc[:, :, 0][cm].astype(np.float32)
                                # circular mean hue: red straddles the 0/180 wrap
                                ang = np.deg2rad(hs * 2.0)
                                hmean = (np.rad2deg(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) / 2.0) % 180
                                if hmean < 12 or hmean > 168:   col = (0, 0, 255)      # red
                                elif hmean < 38:                col = (0, 220, 255)    # amber
                                else:                           col = (0, 225, 70)     # green
                                a2 = lst2[j, cv2.CC_STAT_AREA]
                                r = max(2, int(round((a2 / np.pi) ** 0.5)))
                                cv2.circle(fr, (int(cen2[j][0]), int(cen2[j][1])), r, col, -1, cv2.LINE_AA)
                                halo = np.zeros(fr.shape, np.float32)
                                cv2.circle(halo, (int(cen2[j][0]), int(cen2[j][1])), int(r * 1.9), col, -1, cv2.LINE_AA)
                                fr = fr + cv2.GaussianBlur(halo, (0, 0), 9) * GLOW
                        else:
                            fr = fr + cv2.GaussianBlur(lit[..., None] * src, (0, 0), 7) * GLOW
                        lamps += 1
                fr = np.clip(fr, 0, 255).astype(np.uint8)
                hit += 1
    vw.write(fr); i += 1
vw.release()
print('wrote %s  (%d frames, TL composited in %d, lit lamp seen in %d)' % (OUT, i, hit, lamps))

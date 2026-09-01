"""Temporal stabilization without the ghost trails.

WHAT WAS WRONG WITH v1. It warps the previous frame by optical flow and blends -- sound idea --
but it feeds the BLEND back as the next iteration's history:

    temporal = wgt*cur + (1-wgt)*((1-alpha)*cur + alpha*warp)
    ps = temporal            # <-- IIR feedback

At the alpha=0.9 the chain uses, that is a recursive filter with ~10-frame memory (1/(1-alpha)).
Farneback flow is unreliable on fast motion and flat texture, so wherever it mistracks, the
error is not corrected -- it is re-warped and re-blended for ten more frames. The visible result
is a car-shaped outline sliding across the frame with no car in it.

WHAT THIS CHANGES.
  1. FIR, not IIR. History is the previous ACTUAL frame, never the previous blend, so a flow
     error lives for one frame instead of ten. This alone removes most of the trailing.
  2. Forward-backward flow consistency. Flow is computed both ways; where they disagree by more
     than fb_thresh pixels the warp is untrustworthy and the pixel falls back to the current
     frame. This catches the mistracks the brightness-difference guard misses -- a wrong match
     onto similar-looking road has a SMALL colour difference but a LARGE flow inconsistency,
     which is exactly the case v1's occ_lo=12 sailed past.
  3. Gentler default alpha (0.6, was 0.9) -- with feedback removed, less is needed.
  4. Motion-aware alpha: pixels the flow says are moving fast get less blending, because that
     is where warping is least reliable and ghosting is most visible.

It also reports a GHOST metric, not just flicker, since v1 optimised flicker into a ghosting
problem. ghost = mean |out - cur| inside high-motion regions; it measures how much of the output
is history rather than the present frame. Low flicker with high ghost is the failure v1 had.

Usage: stabilize_frames_v2.py --frames_dir D --out out.mp4 [--alpha 0.6] [--detail_sigma 3]
                              [--fb_thresh 2.0] [--saturation 1.0]
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from vidcodec import fourcc_for   # .avi -> lossless FFV1, anything else -> mp4v
import cv2, glob, argparse, os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames_dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--alpha', type=float, default=0.6)
    ap.add_argument('--detail_sigma', type=float, default=3.0)
    ap.add_argument('--fb_thresh', type=float, default=2.0, help='px of fwd/bwd flow disagreement tolerated')
    ap.add_argument('--motion_ref', type=float, default=6.0, help='px/frame at which blending is fully suppressed')
    ap.add_argument('--saturation', type=float, default=1.0, help='<1 tones down vividness')
    ap.add_argument('--fps', type=int, default=30)
    a = ap.parse_args()

    fs = sorted(glob.glob(os.path.join(a.frames_dir, '*_synthesized_image.jpg')))
    if not fs:
        fs = sorted(glob.glob(os.path.join(a.frames_dir, '*.png')))
    if not fs:
        raise SystemExit('no frames in ' + a.frames_dir)
    h, w = cv2.imread(fs[0]).shape[:2]
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    gx = gx.astype(np.float32); gy = gy.astype(np.float32)

    def desat(img):
        if a.saturation == 1.0:
            return img
        hsv = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= a.saturation
        return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    vw = cv2.VideoWriter(a.out, fourcc_for(a.out), a.fps, (w, h))
    prev = cv2.imread(fs[0]).astype(np.float32)
    prev_out = desat(prev)
    vw.write(np.clip(prev_out, 0, 255).astype(np.uint8))
    pg = cv2.cvtColor(prev.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    pstab = cv2.cvtColor(np.clip(prev_out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)

    raw_d, st_d, ghost, rejected = [], [], [], []
    FB = dict(pyr_scale=0.5, levels=4, winsize=25, iterations=5, poly_n=7, poly_sigma=1.5, flags=0)
    for i in range(1, len(fs)):
        cur = cv2.imread(fs[i]).astype(np.float32)
        cg = cv2.cvtColor(cur.astype(np.uint8), cv2.COLOR_BGR2GRAY)

        fl = cv2.calcOpticalFlowFarneback(cg, pg, None, **FB)        # cur -> prev
        bl = cv2.calcOpticalFlowFarneback(pg, cg, None, **FB)        # prev -> cur

        # forward-backward consistency: follow the flow there and back; a good match returns home
        bx = cv2.remap(bl[..., 0], gx + fl[..., 0], gy + fl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        by = cv2.remap(bl[..., 1], gx + fl[..., 0], gy + fl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        fb_err = np.sqrt((fl[..., 0] + bx) ** 2 + (fl[..., 1] + by) ** 2)
        trust = np.clip(1.0 - fb_err / max(a.fb_thresh, 1e-6), 0, 1)

        # fast-moving pixels get less history, because that is where warping fails worst
        mag = np.sqrt(fl[..., 0] ** 2 + fl[..., 1] ** 2)
        motion_scale = np.clip(1.0 - mag / max(a.motion_ref, 1e-6), 0, 1)

        # FIR: warp the previous ACTUAL frame, never the previous blend
        warp = cv2.remap(prev, gx + fl[..., 0], gy + fl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        eff = (a.alpha * trust * motion_scale)[..., None]
        temporal = (1 - eff) * cur + eff * warp

        if a.detail_sigma > 0:
            out = (cv2.GaussianBlur(temporal, (0, 0), a.detail_sigma)
                   + (cur - cv2.GaussianBlur(cur, (0, 0), a.detail_sigma)))
        else:
            out = temporal
        out = desat(out)
        ou = np.clip(out, 0, 255).astype(np.uint8)
        og = cv2.cvtColor(ou, cv2.COLOR_BGR2GRAY)
        vw.write(ou)

        raw_d.append(np.abs(cg.astype(float) - pg.astype(float))[h // 2:].mean())
        st_d.append(np.abs(og.astype(float) - pstab.astype(float))[h // 2:].mean())
        mv = mag > 1.5
        ghost.append(float(np.abs(ou.astype(float) - cur).mean(2)[mv].mean()) if mv.sum() > 500 else 0.0)
        rejected.append(float((trust < 0.5).mean()) * 100)

        prev = cur; pg = cg; pstab = og
    vw.release()
    r, s = float(np.mean(raw_d)), float(np.mean(st_d))
    print('wrote %s' % a.out)
    print('  ground flicker raw=%.2f stab=%.2f (-%.0f%%)   GHOST=%.2f   flow rejected %.1f%% of px'
          % (r, s, 100 * (r - s) / r, float(np.mean(ghost)), float(np.mean(rejected))))


if __name__ == '__main__':
    main()

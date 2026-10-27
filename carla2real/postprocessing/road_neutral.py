#!/usr/bin/env python3
"""Take the olive cast off the ground (road, sidewalk, curb, markings): hue only, brightness untouched.

WHY. Zoomed road crops, 2026-10-18: real asphalt is neutral grey (Lab a,b ~ 128,127.5 over the licensed real corpus);
the render's road is olive/khaki -- the largest surface in the frame. Earlier grades that also moved brightness
(grade_to_real) cost CIPO and shifted VaVAM plans; this moves only a,b of each ground class's mean to the real target.
  usage: road_neutral.py <in.mp4> <label_dir> <out>   env: STRENGTH=0.8
"""
import glob, json, os, sys
import cv2, numpy as np
from PIL import Image
from carla2real.common.vidcodec import fourcc_for
S = float(os.environ.get('STRENGTH', '0.8'))
T = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'real_targets.json')))
GROUND = [int(x) for x in os.environ.get('GROUND', '13,15,2,24,23').split(',')]   # +17,6,3,4 also neutralises buildings, walls, fences


def process(im, l):
    A = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32); sh = np.zeros_like(A); acc = np.zeros(A.shape[:2], np.float32)
    for c in GROUND:
        m = l == c
        if m.sum() < 800 or str(c) not in T: continue
        d = np.zeros(3, np.float32); d[1:] = np.array(T[str(c)][0], np.float32)[1:] - A[m].mean(0)[1:]
        w = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.5); sh += w[..., None] * d; acc += w
    A += S * sh / np.maximum(acc, 1)[..., None]
    return cv2.cvtColor(np.clip(np.round(A), 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


if __name__ == '__main__':
    IN, LABD, OUT = sys.argv[1:4]
    labs = sorted(glob.glob(os.path.join(LABD, '*.png'))); cap = cv2.VideoCapture(IN); W, H = int(cap.get(3)), int(cap.get(4))
    vw = cv2.VideoWriter(OUT, fourcc_for(OUT), cap.get(5) or 30, (W, H)); n = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        if n < len(labs):
            l = np.array(Image.open(labs[n])); l = l if l.ndim == 2 else l[..., 0]
            f = process(f, cv2.resize(l, (W, H), interpolation=cv2.INTER_NEAREST))
        vw.write(f); n += 1
    vw.release(); print(f'wrote {OUT}: {n} frames (ground hue neutralised, strength {S})')

#!/usr/bin/env python3
"""Smooth the render's fine grain while keeping edges (bilateral filter): cars, lane lines and wall edges stay sharp.

WHY. 2026-10-29: VaVAM plans closest to CARLA on the most-compressed chain and Vision Pilot detects best on the least --
compression softens the render's fine grain. Doing that softening deliberately (bilateral, edge-preserving) on Town03
improved BOTH: CIPO .895 -> .899, VaVAM plan agreement 1.43 -> 1.19 (lossless test, d 7, sigma 35).
  usage: degrain.py <in.mp4> <out>   env: DIAM=7 SIG=35
"""
import os, sys
import cv2

from carla2real.common.vidcodec import fourcc_for
DI, SIG = int(os.environ.get('DIAM', '7')), float(os.environ.get('SIG', '35'))
IN, OUT = sys.argv[1:3]
c = cv2.VideoCapture(IN); W, H = int(c.get(3)), int(c.get(4)); w = cv2.VideoWriter(OUT, fourcc_for(OUT), c.get(5) or 30, (W, H)); n = 0
while True:
    ok, f = c.read()
    if not ok: break
    w.write(cv2.bilateralFilter(f, DI, SIG, SIG)); n += 1
w.release(); print(f'wrote {OUT}: {n} frames (degrain d {DI} sigma {SIG})')

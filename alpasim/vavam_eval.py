#!/usr/bin/env python3
"""Open-loop driving-policy evaluation of carla2real video with VaVAM (AlpaSim's default driver).

THE QUESTION. Does a driving policy trained on real cameras behave on our video the way it should?
CIPO recall only asks whether the closest car is detected. This asks whether a real-world driver
would plan the right motion from these frames.

HOW. VaVAM sees 8 frames at 2 Hz and a drive command, and plans an ego trajectory at 2 Hz. For each
decision point in a clip:
  * longitudinal error: planned travel at 1, 2 and 3 s against the distance CARLA's car actually
    covered, integrated from the recorded per-frame speeds (frame_speed.txt). Ground truth.
  * lateral jitter: how far the planned 3 s endpoint moves sideways between consecutive decisions.
    No ground truth needed; a policy that keeps changing its mind is being confused by its input.
Every render of the same drive is fed the same decision points and the same command, so the numbers
compare renders, not drives.

The command is STRAIGHT throughout: the recordings carry no route. That is wrong at turns, but wrong
identically for every render, so it does not favour any of them. Lateral error against a real path
needs ego poses, which the recorder now logs (frame_pose.txt) for future recordings.

  usage: vavam_eval.py <town> <tag> [<tag> ...]     tag "CARLA" means the raw simulator frames
"""
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

from alpasim_driver.models.vam_model import VAMModel
from alpasim_driver.models.base import DriveCommand

torch.set_num_threads(int(os.environ.get('THREADS', '16')))
FPS = 30
STEP = 15                      # 2 Hz decision points / context spacing
CTX = 8
HORIZON_S = (1.0, 2.0, 3.0)
DEST = os.environ.get('CARLA2REAL_OUT', 'output')                       # delivered *_FINAL_1920_visionpilot.mp4
REC = os.environ.get('CARLA2REAL_DATA', 'datasets') + '/recorded_{TT}_{W}_inst'   # CARLA rgb/ + frame_speed.txt
OUT = os.environ.get('OUT', 'alpasim/results')
VAM = os.environ.get('VAVAM_DIR', 'alpasim/vavam')                   # VaVAM-B checkpoint + tokenizer

TOWN = sys.argv[1]
WEATHER = os.environ.get('WEATHER', 'sunny')          # 'night' scores the night recording
KEY = TOWN if WEATHER == 'sunny' else f'{TOWN}_{WEATHER}'   # results file prefix
TAGS = sys.argv[2:]
TT = {'town05': 'Town05', 'town10hd': 'Town10HD', 'town03': 'Town03', 'town04': 'Town04',
      'town06': 'Town06'}[TOWN]
rec = REC.format(TT=TT, W=WEATHER)
speeds = np.array([float(x) for x in open(f'{rec}/frame_speed.txt').read().split()])
cum = np.concatenate([[0.0], np.cumsum(speeds / FPS)])          # metres travelled up to frame i

model = VAMModel(checkpoint_path=f'{VAM}/VAM_width_1024_pretrained_139k.pt',
                 tokenizer_path=f'{VAM}/VQ_ds16_16384_llamagen_encoder.jit',
                 device=torch.device('cuda' if os.environ.get('GPU') == '1' else 'cpu'),
                 camera_ids=['cam'], context_length=CTX)
if os.environ.get('GPU') != '1':
    # VAMModel hands float16 to its action head unconditionally; that only works under CUDA autocast
    model.DTYPE = torch.float32


class Inp:                                                       # the fields VAMModel reads
    def __init__(self, frames):
        self.camera_images = {'cam': frames}
        self.command = DriveCommand.STRAIGHT


def load_frames(tag, n):
    if tag == 'CARLA':
        # CARLA PNGs are stored BGR-as-RGB, so cv2's read order IS real RGB
        return lambda i: cv2.imread(f'{rec}/rgb/{i:06d}.png')
    cap = cv2.VideoCapture(f'{DEST}/{TOWN}_{WEATHER}_vp55_{tag}_FINAL_1920_visionpilot.mp4')
    cache = {}
    def get(i):
        if i not in cache:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, f = cap.read()
            cache[i] = cv2.cvtColor(f, cv2.COLOR_BGR2RGB) if ok else None
        return cache[i]
    return get


n = len(speeds)
points = list(range(STEP * (CTX - 1), n - int(HORIZON_S[-1] * FPS) - 1, STEP))
if os.environ.get('LIMIT'):
    points = points[:int(os.environ['LIMIT'])]
os.makedirs(OUT, exist_ok=True)
summary = {}
for tag in TAGS:
    get = load_frames(tag, n)
    t0 = time.time()
    plans = []
    for p in points:
        frames = [(k, get(p - (CTX - 1 - k) * STEP)) for k in range(CTX)]
        if any(f is None for _, f in frames):
            continue
        pred = model.predict(Inp(frames))
        xy = np.asarray(pred.candidate_positions[pred.selected_index][:, :2], dtype=float)   # rig frame
        plans.append((p, xy))
    # forward axis = the one that grows along the plan
    fw = int(np.argmax(np.abs(plans[0][1][-1]))) if plans else 0
    lat = 1 - fw
    long_err = {h: [] for h in HORIZON_S}
    for p, xy in plans:
        for h in HORIZON_S:
            k = int(round(h * 2)) - 1                             # 2 Hz waypoints: 1 s -> index 1
            if k < len(xy):
                actual = cum[min(n, p + int(h * FPS))] - cum[p]
                long_err[h].append(abs(abs(xy[k][fw]) - actual))
    ends = np.array([xy[-1][lat] for _, xy in plans])
    jitter = float(np.mean(np.abs(np.diff(ends)))) if len(ends) > 1 else float('nan')
    summary[tag] = {'decisions': len(plans), 'jitter_m': jitter,
                    **{f'long_err_{h:g}s_m': float(np.mean(v)) for h, v in long_err.items()},
                    'seconds': round(time.time() - t0)}
    print(tag, json.dumps(summary[tag]), flush=True)
    np.save(f'{OUT}/{KEY}_{tag}_plans.npy', np.array([np.concatenate([[p], xy.ravel()]) for p, xy in plans]))

# merge: a later run for new tags must not drop the earlier tags' rows
sp = f'{OUT}/{KEY}_summary.json'
merged = json.load(open(sp)) if os.path.exists(sp) else {}
merged.update(summary)
json.dump(merged, open(sp, 'w'), indent=1)
print(f'\n{KEY}: VaVAM-B open loop, {len(points)} decision points at 2 Hz, command STRAIGHT')
print(f'{"render":10s} {"long 1s":>8s} {"long 2s":>8s} {"long 3s":>8s} {"lat jitter":>11s}')
for t, s in summary.items():
    print(f'{t:10s} {s["long_err_1s_m"]:8.2f} {s["long_err_2s_m"]:8.2f} {s["long_err_3s_m"]:8.2f} {s["jitter_m"]:11.2f}')

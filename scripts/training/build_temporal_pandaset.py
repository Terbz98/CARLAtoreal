#!/usr/bin/env python3
"""Licensed SEQUENCE corpus for self-consistency training: PandaSet front camera, daytime scenes, consecutive frames.

pandaset_stage/train_* holds pshr_00000..08239 = 103 scenes x 80 frames, sorted scene-then-frame (build_pandaset_corpus.sh),
with every conditioning channel. Night/dusk scenes are dropped (mean grey < DAY_MIN on the scene's middle frame). Frames
are linked as s<scene>_<frame>.* so TemporalDataset groups them by scene; flow t -> t-1 (backward, Farneback on 512-wide
grey, float16) as gen_temporal_data.py does, so the loader's convention is unchanged.
"""
import os, cv2, numpy as np
from concurrent.futures import ProcessPoolExecutor
B = os.environ.get('CARLA2REAL_DATA', 'datasets'); SRC = f'{B}/pandaset_stage'; DST = f'{B}/training_temporal_pandaset'
CH = {'train_img': '.jpg', 'train_label': '.png', 'train_edge': '.png', 'train_depth': '.png', 'train_normal': '.png',
      'train_chroma': '.png', 'train_texture': '.png'}
DAY_MIN = float(os.environ.get('DAY_MIN', '70'))
for c in list(CH) + ['train_flow']: os.makedirs(f'{DST}/{c}', exist_ok=True)


def grey(i):
    return cv2.cvtColor(cv2.resize(cv2.imread(f'{SRC}/train_img/pshr_{i:05d}.jpg'), (512, 288), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)


def scene(s):
    if grey(s * 80 + 40).mean() < DAY_MIN: return 0
    prev = None
    for f in range(80):
        i = s * 80 + f; stem = f's{s:03d}_{f:06d}'
        for c, ext in CH.items():
            src = os.path.realpath(f'{SRC}/{c}/pshr_{i:05d}{ext}'); d = f'{DST}/{c}/{stem}{ext}'
            if not os.path.lexists(d): os.symlink(src, d)
        g = grey(i)
        if prev is not None:
            np.save(f'{DST}/train_flow/{stem}.npy', cv2.calcOpticalFlowFarneback(g, prev, None, 0.5, 3, 21, 3, 5, 1.2, 0).astype(np.float16))
        prev = g
    return 1


with ProcessPoolExecutor(16) as ex: kept = sum(ex.map(scene, range(103)))
print(f'kept {kept} daytime scenes -> {kept * 80} frames in {DST}')

#!/usr/bin/env python3
"""Per-class flicker relative to CARLA: low-passed temporal change after motion compensation, with the motion taken
from CARLA's frames (true motion). CARLA's own value is the floor. usage: flicker_vs_carla.py <Town> <start> <n> name=video ..."""
import sys, cv2, numpy as np
from PIL import Image
TT, S, N = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
R = f'datasets/recorded_{TT}_sunny_inst/rgb'; L = f'datasets/training_v12_mapillary/test_{TT}_sunny_inst_gt_label'
def frames(src):
    if src == 'CARLA': return [cv2.resize(np.ascontiguousarray(cv2.imread(f'{R}/{i:06d}.png')[:, :, ::-1]), (960, 480)) for i in range(S, S + N)]
    c = cv2.VideoCapture(src); c.set(1, S); return [cv2.resize(c.read()[1], (960, 480)) for _ in range(N)]
car = frames('CARLA'); g = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in car]
flows = [None] + [cv2.calcOpticalFlowFarneback(g[k], g[k - 1], None, 0.5, 4, 21, 3, 5, 1.2, 0) for k in range(1, N)]
gx, gy = np.meshgrid(np.arange(960, dtype=np.float32), np.arange(480, dtype=np.float32))
labs = [cv2.resize((lambda x: x if x.ndim == 2 else x[..., 0])(np.array(Image.open(f'{L}/{i:06d}.png'))), (960, 480), interpolation=cv2.INTER_NEAREST) for i in range(S, S + N)]
CL = {'building': [17, 6, 16], 'vegetation': [30, 29], 'road': [13], 'vehicle': [55, 54, 61, 52, 57]}
srcs = [('CARLA', 'CARLA')] + [a.split('=', 1) for a in sys.argv[4:]]
print(f'{"":12s}' + ''.join(f'{k:>12s}' for k in CL))
for name, src in srcs:
    fr = frames(src); acc = {k: [] for k in CL}
    for k in range(1, N):
        p = cv2.cvtColor(fr[k - 1], cv2.COLOR_BGR2GRAY).astype(np.float32); c = cv2.cvtColor(fr[k], cv2.COLOR_BGR2GRAY).astype(np.float32)
        wp = cv2.remap(p, gx + flows[k][..., 0], gy + flows[k][..., 1], cv2.INTER_LINEAR)
        d = np.abs(cv2.GaussianBlur(wp, (0, 0), 2) - cv2.GaussianBlur(c, (0, 0), 2))
        for kk, ids in CL.items():
            m = cv2.erode(np.isin(labs[k], ids).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            if m.sum() > 500: acc[kk].append(d[m].mean())
    print(f'{name:12s}' + ''.join(f'{np.mean(v):12.2f}' if v else f'{"-":>12s}' for v in acc.values()))

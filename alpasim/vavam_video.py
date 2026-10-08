#!/usr/bin/env python3
"""Video of the VaVAM open-loop eval: 2x2 grid (CARLA, v50m, v85d, v90dv) with each render's planned
3 s path drawn on the road, and planned vs actually-driven distance as text.

Camera from record_town_auto.py --fov 55 --pitch -5: 2.0 m ahead of the car origin, 1.5 m up.
The path is the plan made at the latest 2 Hz decision point, held for the next 15 frames.
  usage: vavam_video.py <town> [<weather>:<tag> x4]   e.g. town10hd sunny:CARLA sunny:v90dv night:CARLA night:v93nw
"""
import os, sys, math, cv2, numpy as np
TOWN = sys.argv[1]
TT = {'town05': 'Town05', 'town10hd': 'Town10HD', 'town03': 'Town03', 'town04': 'Town04', 'town06': 'Town06'}[TOWN]
# tiles: "<weather>:<tag>" (default: four sunny renders); each weather has its own CARLA drive
TILES = [x.split(':') for x in (sys.argv[2:] or ['sunny:CARLA', 'sunny:v50m', 'sunny:v85d', 'sunny:v90dv'])]
DEST = os.environ.get('CARLA2REAL_OUT', 'output')
REC = os.environ.get('CARLA2REAL_DATA', 'datasets') + '/recorded_{TT}_{W}_inst'
RES = os.environ.get('OUT', 'alpasim/results')
PAL = [(255, 255, 255), (60, 60, 255), (200, 200, 200), (255, 170, 60)]
W, H, FOV, PITCH, CAM_X, CAM_Z = 960, 480, 55.0, -5.0, 2.0, 1.5
f = (W / 2) / math.tan(math.radians(FOV / 2)); cp, sp_ = math.cos(math.radians(PITCH)), math.sin(math.radians(PITCH))
def drive(w):
    sp = np.array([float(x) for x in open(REC.format(TT=TT, W=w) + '/frame_speed.txt').read().split()])
    return np.concatenate([[0.0], np.cumsum(sp / 30)]), len(sp)
DRIVES = {w: drive(w) for w in {w for w, _ in TILES}}
n = min(d[1] for d in DRIVES.values())

def project(x, y):
    X, Y, Z = x - CAM_X, y, -CAM_Z
    fwd = X * cp + Z * sp_; up = -X * sp_ + Z * cp
    if fwd < 0.5: return None
    return int(W / 2 - f * Y / fwd), int(H / 2 - f * up / fwd)

def key(w):
    return TOWN if w == 'sunny' else f'{TOWN}_{w}'
plans, caps = {}, {}
for w, t in TILES:
    a = np.load(f'{RES}/{key(w)}_{t}_plans.npy')
    plans[(w, t)] = {int(r[0]): r[1:].reshape(-1, 2) for r in a}
    if t != 'CARLA':
        caps[(w, t)] = cv2.VideoCapture(f'{DEST}/{TOWN}_{w}_vp55_{t}_FINAL_1920_visionpilot.mp4')
name = '_'.join(f'{w}-{t}' for w, t in TILES)
out = f'{RES}/{TOWN}_vavam_{name}.avi'
vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'MJPG'), 30, (2 * W, 2 * H))
for i in range(n):
    tiles = []
    for j, (w, t) in enumerate(TILES):
        rec = REC.format(TT=TT, W=w); cum = DRIVES[w][0]; col = PAL[j % len(PAL)]
        if t == 'CARLA':
            img = cv2.cvtColor(cv2.imread(f'{rec}/rgb/{i:06d}.png'), cv2.COLOR_RGB2BGR)
        else:
            ok, img = caps[(w, t)].read()
            if not ok: img = np.zeros((H, W, 3), np.uint8)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        pl = plans[(w, t)]
        d = max([p for p in pl if p <= i], default=None)
        txt2 = 'waiting for 8 frames of context'
        if d is not None and i < d + 15 + 1:
            xy = pl[d]
            pts = [project(0.0, 0.0)] + [project(x, y) for x, y in xy]
            pts = [p for p in pts if p is not None]
            for a_, b_ in zip(pts, pts[1:]):
                cv2.line(img, a_, b_, col, 5, cv2.LINE_AA)
            for p in pts[1:]:
                cv2.circle(img, p, 7, col, -1, cv2.LINE_AA)
            actual = cum[min(len(cum) - 1, d + 90)] - cum[d]
            txt2 = f'plans {abs(xy[5][0]):4.1f} m in 3 s | actually drove {actual:4.1f} m'
        cv2.rectangle(img, (0, 0), (W, 62), (0, 0, 0), -1)
        lab = f'{w}: ' + ('CARLA (raw)' if t == 'CARLA' else t)
        cv2.putText(img, lab, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, .8, col, 2, cv2.LINE_AA)
        cv2.putText(img, txt2, (12, 52), cv2.FONT_HERSHEY_SIMPLEX, .62, (230, 230, 230), 1, cv2.LINE_AA)
        tiles.append(img)
    frame = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4])])
    cv2.putText(frame, f'{TT}  t={i/30:5.1f}s  VaVAM-B (AlpaSim driver), command STRAIGHT',
                (W - 330, 2 * H - 14), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2, cv2.LINE_AA)
    vw.write(frame)
vw.release(); print(out)

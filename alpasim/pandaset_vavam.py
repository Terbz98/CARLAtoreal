#!/usr/bin/env python3
"""VaVAM (AlpaSim's driver) on REAL driving video: PandaSet front camera, scored against the path actually driven.

WHY. User, 2026-10-27: "try comparing with real life videos and run them in alpasim, make sure the accuracy is the same
if not better". Our CARLA scores say how VaVAM's plan on a render compares with its plan on raw CARLA, and with the
distance CARLA's car actually drove (vavam_eval.py long_err). This gives the reference: how VaVAM behaves on real camera
footage -- its error profile, bias (plans too short or too long) and jitter -- so a render can be judged by whether the
policy behaves on it the way it behaves on the real world.

PandaSet: 103 sequences, 8 s at 10 Hz, front camera 1920x1080 (HFOV ~52 deg), camera poses (poses.json, world frame,
quaternion w,x,y,z, camera axes x right, y down, z forward). VaVAM sees 8 frames at 2 Hz (5-frame spacing), plans 6
waypoints at 2 Hz (3 s). Decision points: frames 35, 40, 45 (history and 3 s future both inside the sequence).
Ground truth: future camera positions in the camera frame at the decision point; forward = z, lateral = x.
COMMAND=straight (as in the CARLA runs, comparable) or gt (from the true 3 s heading change: >8 deg left/right).
  usage: pandaset_vavam.py     env: COMMAND=straight|gt  SEQS=<n>  OUT=results/pandaset_<cmd>.json
"""
import glob, json, os, sys, time
import cv2, numpy as np, torch
from alpasim_driver.models.vam_model import VAMModel
from alpasim_driver.models.base import DriveCommand

torch.set_num_threads(int(os.environ.get('THREADS', '16')))
PS = os.environ.get('PANDASET', 'data/pandaset')
VAM = os.environ.get('VAVAM_DIR', 'vavam')
CMD = os.environ.get('COMMAND', 'straight')
OUT = os.environ.get('OUT', f'results/pandaset_{CMD}.json')
STEP, CTX, POINTS = 5, 8, (35, 40, 45)

model = VAMModel(checkpoint_path=f'{VAM}/VAM_width_1024_pretrained_139k.pt',
                 tokenizer_path=f'{VAM}/VQ_ds16_16384_llamagen_encoder.jit',
                 device=torch.device('cpu'), camera_ids=['cam'], context_length=CTX)
model.DTYPE = torch.float32


class Inp:
    def __init__(self, frames, cmd):
        self.camera_images = {'cam': frames}; self.command = cmd


def rot(q):
    w, x, y, z = q['w'], q['x'], q['y'], q['z']
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


seqs = sorted(d for d in os.listdir(PS) if os.path.exists(f'{PS}/{d}/camera/front_camera/poses.json'))
if os.environ.get('SEQS'): seqs = seqs[:int(os.environ['SEQS'])]
rows = []; t0 = time.time()
for s in seqs:
    cam = f'{PS}/{s}/camera/front_camera'
    poses = json.load(open(f'{cam}/poses.json'))
    imgs = sorted(glob.glob(f'{cam}/*.jpg'))
    if len(imgs) < POINTS[-1] + 31: continue
    for p in POINTS:
        R0 = rot(poses[p]['heading']); p0 = np.array([poses[p]['position'][k] for k in 'xyz'])
        fut = []
        for h in range(1, 7):                                             # 0.5 s .. 3 s
            pw = np.array([poses[p + 5 * h]['position'][k] for k in 'xyz'])
            c = R0.T @ (pw - p0); fut.append((c[2], c[0]))                # (forward, lateral right)
        fut = np.array(fut)
        if np.linalg.norm(fut[-1]) < 1.0: continue                       # standing still: nothing to plan
        ang = np.degrees(np.arctan2(fut[-1][1], fut[-1][0]))
        cmd = DriveCommand.STRAIGHT
        if CMD == 'gt' and abs(ang) > 8: cmd = DriveCommand.RIGHT if ang > 0 else DriveCommand.LEFT
        frames = [(k, cv2.cvtColor(cv2.imread(imgs[p - (CTX - 1 - k) * STEP]), cv2.COLOR_BGR2RGB)) for k in range(CTX)]
        pred = model.predict(Inp(frames, cmd))
        xy = np.asarray(pred.candidate_positions[pred.selected_index][:, :2], dtype=float)
        rows.append({'seq': s, 'p': p, 'plan': xy.tolist(), 'true': fut.tolist(), 'cmd': int(cmd)})
    print(f'{s}: {len(rows)} decisions, {time.time() - t0:.0f} s', flush=True)

# plan axes: forward = the axis that grows; lateral sign fixed once for the whole set (the one that fits better)
P = np.array([r['plan'] for r in rows]); G = np.array([r['true'] for r in rows])
fw = int(np.argmax(np.abs(P[:, -1]).mean(0))); lat = 1 - fw
pf, pl = np.abs(P[..., fw]), P[..., lat]
sgn = 1 if np.abs(pl - G[..., 1]).mean() <= np.abs(-pl - G[..., 1]).mean() else -1
pl = sgn * pl
res = {'decisions': len(rows), 'command': CMD}
for h, k in ((1, 1), (2, 3), (3, 5)):
    res[f'long_err_{h}s_m'] = float(np.abs(pf[:, k] - G[:, k, 0]).mean())
    res[f'long_bias_{h}s_m'] = float((pf[:, k] - G[:, k, 0]).mean())          # <0 = plans too short
    res[f'lat_err_{h}s_m'] = float(np.abs(pl[:, k] - G[:, k, 1]).mean())
    res[f'ade_{h}s_m'] = float(np.hypot(pf[:, :k + 1] - G[:, :k + 1, 0], pl[:, :k + 1] - G[:, :k + 1, 1]).mean())
res['true_3s_m'] = float(G[:, 5, 0].mean()); res['plan_3s_m'] = float(pf[:, 5].mean())
json.dump({'summary': res, 'rows': rows}, open(OUT, 'w'))
print(json.dumps(res, indent=1))

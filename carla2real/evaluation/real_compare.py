#!/usr/bin/env python3
"""Render vs REAL photographs, per class: statistics and zoomed crop galleries.

User, 2026-10-15: "compare your results with real-life cars, zoom in, does it look correct, compare the buildings with
real-life buildings, does it look legit". Real side: ZOD (CC BY-SA) and PandaSet (CC BY) frames with their Mapillary-65
labels from the training corpus. Render side: delivered clips of a tag on the five towns with CARLA's labels.

Per class (car, building, vegetation, road, sidewalk): Lab mean/std, chroma, fine texture (|L - blur1.5|), mid texture
(|blur1.5 - blur6|), highlight spread (p98 - p50 of L), edge density (Canny), all on eroded class masks, at 1024x512.
Galleries: the 12 largest class components per side, cropped to their bounding box (+10%), resized to 256 px tall.
  usage: real_compare.py <tag> [<tag> ...]      -> stats table on stdout, galleries in $OUT/<class>_<tag>.jpg
"""
import glob, os, random, sys
import cv2, numpy as np
from PIL import Image

B = os.environ.get('CARLA2REAL_ROOT', '.'); D = os.environ.get('CARLA2REAL_DATA', 'data')
OUT = os.environ.get('OUT', 'results/real_compare'); os.makedirs(OUT, exist_ok=True)
CLASSES = {'car': [55], 'building': [17], 'vegetation': [30], 'road': [13], 'sidewalk': [15]}
TOWNS = [('town05', 'Town05'), ('town10hd', 'Town10HD'), ('town03', 'Town03'), ('town04', 'Town04'), ('town06', 'Town06')]
random.seed(0)
WX = os.environ.get('WEATHER', 'sunny')   # night: real = ZOD/PandaSet night frames, render = night clips
REALGLOBS = {'sunny': ['training_v98/train_img/zshr_*.jpg', 'training_v98/train_img/p*hr_*.jpg'],
             'night': ['training_v97n/train_img/zngt_*', 'training_v97n/train_img/psngt_*']}[WX]


def feats(im, m):
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32); L = lab[..., 0]
    mm = cv2.erode(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    if mm.sum() < 1500: return None
    b1 = cv2.GaussianBlur(L, (0, 0), 1.5); b6 = cv2.GaussianBlur(L, (0, 0), 6)
    ed = cv2.Canny(im, 60, 150) > 0
    v = L[mm]
    return [v.mean(), v.std(), np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)[mm].mean(),
            np.abs(L - b1)[mm].mean(), np.abs(b1 - b6)[mm].mean(), np.percentile(v, 98) - np.percentile(v, 50), ed[mm].mean() * 100]


def crops(im, m, k=3):
    n, cc, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    out = []
    for i in sorted(range(1, n), key=lambda i: -st[i, 4])[:k]:
        x, y, w, h, a = st[i]
        if a < 4000: continue
        px, py = int(w * 0.1), int(h * 0.1)
        c = im[max(0, y - py):y + h + py, max(0, x - px):x + w + px]
        out.append((a, cv2.resize(c, (max(32, int(c.shape[1] * 256 / max(c.shape[0], 1))), 256))))
    return out


def real_samples(n=300):
    fs = sum((sorted(glob.glob(f'{B}/datasets/{g}')) for g in REALGLOBS), [])
    for f in random.sample(fs, n):
        lf = f.replace('train_img', 'train_label').rsplit('.', 1)[0] + '.png'
        if not os.path.exists(lf): continue
        im = cv2.resize(cv2.imread(f), (1024, 512), interpolation=cv2.INTER_AREA)
        l = np.array(Image.open(lf)); l = l if l.ndim == 2 else l[..., 0]
        yield im, cv2.resize(l, (1024, 512), interpolation=cv2.INTER_NEAREST)


def render_samples(tag, step=40):
    for t, TT in TOWNS:
        c = cv2.VideoCapture(f'{D}/{t}_{WX}_vp55_{tag}_FINAL_1920_visionpilot.mp4')
        if not c.isOpened(): continue
        for i in range(0, 1000, step):
            c.set(1, i); ok, im = c.read()
            if not ok: continue
            l = np.array(Image.open(f'{B}/datasets/training_v12_mapillary/test_{TT}_{WX}_inst_gt_label/{i:06d}.png')); l = l if l.ndim == 2 else l[..., 0]
            yield cv2.resize(im, (1024, 512), interpolation=cv2.INTER_AREA), cv2.resize(l, (1024, 512), interpolation=cv2.INTER_NEAREST)


def collect(gen):
    st = {k: [] for k in CLASSES}; gal = {k: [] for k in CLASSES}
    for im, l in gen:
        for k, ids in CLASSES.items():
            m = np.isin(l, ids)
            f = feats(im, m)
            if f: st[k].append(f)
            if k in ('car', 'building', 'vegetation'): gal[k] += crops(im, m)
    return st, gal


def gallery(items, path, n=12):
    items = sorted(items, key=lambda x: -x[0])[::max(1, len(items) // (n * 2))][:n]
    if not items: return
    rows, row, w = [], [], 0
    for _, c in items:
        if w + c.shape[1] > 1800 and row:
            rows.append(np.hstack(row + [np.zeros((256, 1800 - w, 3), np.uint8)])); row, w = [], 0
        row.append(c[:, :1800]); w += min(c.shape[1], 1800)
    if row: rows.append(np.hstack(row + [np.zeros((256, max(0, 1800 - w), 3), np.uint8)])[:, :1800])
    cv2.imwrite(path, np.vstack(rows))


NAMES = ['L mean', 'L std', 'chroma', 'fine tex', 'mid tex', 'hilite', 'edges%']
sets = [('REAL', collect(real_samples()))] + [(t, collect(render_samples(t))) for t in sys.argv[1:]]
for k in CLASSES:
    print(f'\n== {k} ==   ' + ''.join(f'{n:>10s}' for n in NAMES) + '     n')
    for name, (st, gal) in sets:
        v = np.array(st[k]) if st[k] else None
        print(f'{name:12s}' + (''.join(f'{x:10.1f}' for x in v.mean(0)) + f'{len(v):6d}' if v is not None else '  -'))
        if k in gal: gallery(gal[k], f'{OUT}/{k}_{name}{"_night" if WX == "night" else ""}.jpg')
print(f'\ngalleries in {OUT}')

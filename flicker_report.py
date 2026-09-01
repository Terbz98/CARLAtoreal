#!/usr/bin/env python3
"""One table: flicker, sharpness, ghosting and perception for any set of delivered variants.

The three numbers only mean something together. `alt p99` (the 99th percentile of
|2*x[t] - x[t-1] - x[t+1]|) is what the eye reads as flicker -- plain frame-difference does NOT
track it, and said v50 was calmer than v49 when visual review plainly showed the opposite.
Sharpness catches the muddiness a mean-filter causes. Ghost catches smearing, which the other two
actively reward: any stronger temporal median lowers alt and cannot lower per-frame sharpness, so
without ghost every over-smoothed variant looks like a free win. v50e is the worked example --
best flicker of any post-processed variant, and rejected on ghost alone.

Ghost is measured against a REFERENCE variant (default v50b, the un-deshimmered render) on only
those pixels optical flow says actually moved, so static shimmer does not count as smearing.
ghost 6.30 = the v1 stabiliser = visibly wrong on review. Stay under ~5.

IMPORTANT -- ghost is only meaningful WITHIN a model family. It asks "what did post-processing do
to this render", so the reference must be the same generator with less post applied. Comparing a
v54 clip against a v50b reference measures how the two MODELS differ, which is large and has
nothing to do with smearing; printing that number next to a real ghost score invites exactly the
wrong conclusion. Tags outside the reference's family therefore print "-" rather than a figure.

  usage: flicker_report.py <tag> [<tag> ...] [--towns t1,t2] [--ref v50b] [--frames 200]
"""
from config import OUT
import cv2, glob, json, os, sys
import numpy as np

D = OUT
FLAGS = ('--towns', '--ref', '--frames', '--weather')


def opt(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


# strip both the flags AND their values, or a value like "200" is read as a variant tag
_skip = set()
for _f in FLAGS:
    if _f in sys.argv:
        _skip |= {sys.argv.index(_f), sys.argv.index(_f) + 1}
args = [a for i, a in enumerate(sys.argv[1:], 1) if i not in _skip and not a.startswith('--')]


TOWNS = opt('--towns', 'town03,town04,town05,town06,town10hd').split(',')
WEATHER = opt('--weather', 'sunny')
REF = opt('--ref', 'v50b')
N = int(opt('--frames', 200))
TAGS = args or ['v49', 'v50', 'v50d', 'v50e']


def path(town, tag):
    return f'{D}/{town}_{WEATHER}_vp55_{tag}_FINAL_1920_visionpilot.mp4'


def frames(p, n=N):
    if not os.path.exists(p):
        return []
    c = cv2.VideoCapture(p); out = []
    while len(out) < n:
        ok, f = c.read()
        if not ok: break
        out.append(cv2.resize(f, (960, 480)))
    c.release(); return out


def alt_sharp(fr):
    a = np.stack([cv2.cvtColor(cv2.resize(f, (480, 240)), cv2.COLOR_BGR2GRAY).astype(np.float32)
                  for f in fr])
    alt = np.abs(2 * a[1:-1] - a[:-2] - a[2:])
    sh = np.mean([cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() for f in fr])
    return float(np.percentile(alt.mean(axis=0), 99)), float(sh)


def ghost(fr, ref, gr):
    g = []
    for i in range(1, min(len(fr), len(ref)) - 1):
        fl = cv2.calcOpticalFlowFarneback(gr[i-1], gr[i+1], None, 0.5, 3, 21, 3, 5, 1.2, 0)
        mv = (np.linalg.norm(fl, axis=2) * 0.5) > 1.5
        if mv.sum() > 500:
            d = np.abs(fr[i].astype(int) - ref[i].astype(int)).mean(axis=2)
            g.append(float(d[mv].mean()))
    return float(np.mean(g)) if g else float('nan')


def family(tag):
    """Which generator produced this tag. Ghost is only comparable inside one family."""
    if tag.startswith('v50'):
        return 'v50'
    if tag in ('tsun', 'v54'):
        return 'v54'
    if tag in ('tnig', 'v55'):
        return 'v55'
    if tag == 'v56':
        return 'v56'
    if tag in ('v47', 'v47r', 'v51'):
        return 'v51'
    return tag


def cipo(tag, town):
    f = f'{D}/logs_{tag}/{town}_{WEATHER}_score.json'
    if not os.path.exists(f):
        return None, None
    j = json.load(open(f))
    j = j[0] if isinstance(j, list) else j
    return j.get('cipo', {}).get('recall'), j.get('rng', {}).get('mae')


acc = {t: {'alt': [], 'sh': [], 'gh': [], 'ci': [], 'rg': []} for t in TAGS}
for town in TOWNS:
    ref = frames(path(town, REF))
    gr = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in ref] if ref else []
    print(f'=== {town}' + ('' if ref else f'   (no {REF} reference -- ghost unavailable)'))
    print(f'   {"tag":<8}{"alt p99":>9}{"sharp":>9}{"ghost":>8}{"CIPO":>8}{"rng":>7}')
    for tag in TAGS:
        fr = frames(path(town, tag))
        if not fr:
            print(f'   {tag:<8}{"-- not delivered --":>33}')
            continue
        a, s = alt_sharp(fr)
        same = family(tag) == family(REF)
        g = ghost(fr, ref, gr) if (ref and tag != REF and same) else float('nan')
        c, r = cipo(tag, town)
        gs = f'{g:8.2f}' if g == g else '       -'
        print(f'   {tag:<8}{a:9.2f}{s:9.1f}{gs}'
              f'{("%8.3f" % c) if c is not None else "       -"}'
              f'{("%7.2f" % r) if r is not None else "      -"}')
        acc[tag]['alt'].append(a); acc[tag]['sh'].append(s)
        if g == g: acc[tag]['gh'].append(g)
        if c is not None: acc[tag]['ci'].append(c); acc[tag]['rg'].append(r)

# n is printed PER TAG because a tag missing one town would otherwise be averaged over fewer
# towns than the others and silently compared against them. That exactly happened once: a v54
# mean over 4 towns read 0.890 against v50d's 5-town 0.887 and looked like a win, when the
# missing town was v54's worst and the true mean was 0.849.
print(f'\n=== MEAN   (ghost 6.30 = visible trails; stay under ~5)')
print(f'   {"tag":<8}{"alt p99":>9}{"sharp":>9}{"ghost":>8}{"CIPO":>8}{"rng":>7}{"towns":>7}{"scored":>8}')
for tag in TAGS:
    a = acc[tag]
    if not a['alt']:
        continue
    m = lambda v: (sum(v) / len(v)) if v else float('nan')
    gm = m(a["gh"]); gs = f'{gm:8.2f}' if gm == gm else '       -'
    flag = '' if len(a['ci']) in (0, len(a['alt'])) else '  <- INCOMPLETE'
    print(f'   {tag:<8}{m(a["alt"]):9.2f}{m(a["sh"]):9.1f}{gs}'
          f'{m(a["ci"]):8.3f}{m(a["rg"]):7.2f}{len(a["alt"]):7d}{len(a["ci"]):8d}{flag}')

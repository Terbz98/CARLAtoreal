#!/usr/bin/env python3
"""
Score a Vision Pilot run against the CARLA ground truth exported by export_gt.py.

Until now every model choice in this pipeline has been made by eye. The GT jsons carry
per-frame object boxes with metric range/lateral plus road-line points in ego coordinates,
and VP's own run log carries its per-frame plan. This joins the two.

  usage:  score_vp.py GT.json RUN.log [RUN2.log ...] [--json OUT.json] [--label NAME]

VP log line, one per inference frame:
  plan: tyre=.. accel=.. | cte=-0.94m(raw=-0.93m) cte_dot=.. epsi=.. epsi_dot=.. kappa=..
        | cipo=true dist=13.9 m vel=-0.13 m/s

Metrics:
  CIPO range     MAE / bias against the GT closest in-path object, split by distance band
  CIPO presence  recall on frames that have an in-path object, false-alarm rate on frames
                 that do not
  Lane CTE       VP's cte against a lane centre recovered from the GT road-line points
  Stability      frame-to-frame |d cte| -- catches the jitter that a still frame hides
"""
import sys, os, re, json, math
import numpy as np

# An object counts as "in path" if its lateral offset puts it inside the ego lane.
# CARLA lanes are 3.5 m, so half-width 1.75; 1.8 gives a little slack for box noise.
IN_PATH_M   = 1.8
MAX_RANGE_M = 120.0
BANDS = [(0, 15), (15, 30), (30, 60), (60, MAX_RANGE_M)]

PLAN = re.compile(
    r'cte=(?P<cte>[-+0-9.]+)m.*?'
    r'epsi=(?P<epsi>[-+0-9.]+).*?'
    r'kappa=(?P<kappa>[-+0-9.eE]+).*?'
    r'cipo=(?P<cipo>true|false)'
    r'(?:.*?dist=(?P<dist>[-+0-9.]+)\s*m)?'
    r'(?:.*?vel=(?P<vel>[-+0-9.]+)\s*m/s)?'
)


def parse_log(path):
    """Pull the per-frame plan out of a VP run log."""
    out = []
    for line in open(path, errors='ignore'):
        if 'plan:' not in line:
            continue
        m = PLAN.search(line)
        if not m:
            continue
        d = m.groupdict()
        out.append(dict(
            cte=float(d['cte']),
            epsi=float(d['epsi']),
            kappa=float(d['kappa']),
            cipo=d['cipo'] == 'true',
            dist=float(d['dist']) if d['dist'] else None,
            vel=float(d['vel']) if d['vel'] else None,
        ))
    return out


def log_resolution(path):
    """VP rescales H for input larger than its native 1024x512; sx != 1 means a lossy run."""
    for line in open(path, errors='ignore'):
        if 'H_resized' in line:
            raw = re.search(r'raw=(\S+)', line)
            sx = re.search(r'sx=([0-9.]+)', line)
            return (raw.group(1) if raw else '?'), (float(sx.group(1)) if sx else float('nan'))
    return '?', float('nan')


def gt_cipo(frame):
    """Closest in-path object with a usable range, mirroring what VP's CIPO means."""
    best = None
    for o in frame['objects']:
        d, lat = o.get('distance_m'), o.get('lateral_m')
        if d is None or lat is None or d > MAX_RANGE_M or d <= 0:
            continue
        if abs(lat) > IN_PATH_M:
            continue
        if best is None or d < best['distance_m']:
            best = o
    return best


def gt_lane_cte(frame, lo=5.0, hi=25.0, min_side=3):
    """
    Recover the ego lane centre from the GT road-line points and turn it into a CTE.

    lane_points_ego_xy are raw road-line pixels projected to ego coords, not fitted lines,
    so this stays deliberately crude: take the points in a near band ahead, split them by
    side, and take the median of each. Requires markings on BOTH sides -- a one-sided fit
    would silently invent a lane. Returns None when it cannot be trusted.
    """
    pts = frame.get('lane_points_ego_xy')
    if not pts:
        return None
    band = [p for p in pts if lo <= p[0] <= hi]
    left = [p[1] for p in band if p[1] < 0]
    right = [p[1] for p in band if p[1] > 0]
    if len(left) < min_side or len(right) < min_side:
        return None
    # The INNERMOST marking on each side, not the median of that side. lane_points_ego_xy
    # spans every road line in view, so a plain median lands between the ego lane's marking and
    # the next lane's: on town03 sunny it recovered a 6.94 m "lane" and threw away 80% of frames
    # as implausible. Taking the point closest to y=0 on each side recovers 3.48 m against
    # CARLA's true 3.5 m, and lifts usable frames from 195 to 953.
    ly, ry = np.percentile(left, 85), np.percentile(right, 15)
    width = ry - ly
    if not (2.0 <= width <= 5.5):          # not a plausible single lane
        return None
    centre = 0.5 * (ly + ry)
    # ego sits at y=0; CTE is the ego's offset from the centre, so the negated centre offset
    return -centre


def best_offset(gt_frames, plan, probe=(0, 1, 2, -1)):
    """
    VP's AutoDrive consumes (image_prev, image_curr) so its first output lags the clip by a
    frame -- but assuming that is how alignment bugs get baked in. Pick the offset that
    actually correlates best on CIPO range.
    """
    scored = []
    for off in probe:
        errs = []
        for i, p in enumerate(plan):
            j = i + off
            if not (0 <= j < len(gt_frames)):
                continue
            g = gt_cipo(gt_frames[j])
            if g is None or not p['cipo'] or p['dist'] is None:
                continue
            errs.append(abs(p['dist'] - g['distance_m']))
        scored.append((np.mean(errs) if len(errs) > 30 else float('inf'), off, len(errs)))
    scored.sort()
    return scored[0][1], scored


def score(gt_path, log_path):
    gt = json.load(open(gt_path))
    frames = gt['frames']
    plan = parse_log(log_path)
    if not plan:
        raise SystemExit(f'no plan lines in {log_path}')

    off, probes = best_offset(frames, plan)
    raw_res, sx = log_resolution(log_path)

    rng_err, rng_pairs = [], []
    tp = fp = fn = tn = 0
    cte_err, cte_cov = [], 0
    for i, p in enumerate(plan):
        j = i + off
        if not (0 <= j < len(frames)):
            continue
        f = frames[j]
        g = gt_cipo(f)

        # presence
        if g is not None and p['cipo']:
            tp += 1
        elif g is not None and not p['cipo']:
            fn += 1
        elif g is None and p['cipo']:
            fp += 1
        else:
            tn += 1

        # range, only where both agree something is there
        if g is not None and p['cipo'] and p['dist'] is not None:
            rng_err.append(p['dist'] - g['distance_m'])
            rng_pairs.append((g['distance_m'], p['dist']))

        # lane
        t = gt_lane_cte(f)
        if t is not None:
            cte_err.append(p['cte'] - t)
            cte_cov += 1

    rng_err = np.array(rng_err)
    cte_err = np.array(cte_err)
    ctes = np.array([p['cte'] for p in plan])
    jitter = np.abs(np.diff(ctes))

    bands = {}
    for lo, hi in BANDS:
        sel = [(t, v) for t, v in rng_pairs if lo <= t < hi]
        if sel:
            e = np.array([v - t for t, v in sel])
            bands[f'{lo}-{hi}m'] = dict(n=len(sel), mae=float(np.mean(np.abs(e))),
                                        bias=float(np.mean(e)))

    return dict(
        log=os.path.basename(log_path),
        gt=os.path.basename(gt_path),
        input_res=raw_res, h_scale=sx,
        frames=len(plan), offset=off,
        cipo=dict(
            recall=tp / (tp + fn) if tp + fn else None,
            false_alarm=fp / (fp + tn) if fp + tn else None,
            n_gt_present=tp + fn, n_gt_absent=fp + tn),
        rng=dict(n=len(rng_err),
                 mae=float(np.mean(np.abs(rng_err))) if len(rng_err) else None,
                 bias=float(np.mean(rng_err)) if len(rng_err) else None,
                 p95=float(np.percentile(np.abs(rng_err), 95)) if len(rng_err) else None,
                 bands=bands),
        lane=dict(coverage=cte_cov / len(plan),
                  mae=float(np.mean(np.abs(cte_err))) if len(cte_err) else None,
                  bias=float(np.mean(cte_err)) if len(cte_err) else None,
                  p95=float(np.percentile(np.abs(cte_err), 95)) if len(cte_err) else None),
        stability=dict(cte_jitter_mean=float(np.mean(jitter)),
                       cte_jitter_p99=float(np.percentile(jitter, 99)),
                       cte_abs_mean=float(np.mean(np.abs(ctes))),
                       frac_over_1m=float(np.mean(np.abs(ctes) > 1.0))),
    )


def fmt(r):
    def n(x, f='%.2f'):
        return '  n/a' if x is None else f % x
    L = []
    L.append(f"  {r['log']}   vs {r['gt']}")
    warn = '  <-- LOSSY, VP rescales H' if r['h_scale'] and r['h_scale'] > 1.001 else ''
    L.append(f"    input {r['input_res']}  H sx={r['h_scale']:.4f}{warn}")
    L.append(f"    frames {r['frames']}  aligned at offset {r['offset']:+d}")
    c = r['cipo']
    L.append(f"    CIPO   recall {n(c['recall'],'%.3f')} on {c['n_gt_present']} frames"
             f"   false-alarm {n(c['false_alarm'],'%.3f')} on {c['n_gt_absent']}")
    g = r['rng']
    L.append(f"    RANGE  MAE {n(g['mae'])} m   bias {n(g['bias'],'%+.2f')} m"
             f"   p95 {n(g['p95'])} m   (n={g['n']})")
    for k, v in g['bands'].items():
        L.append(f"             {k:<10} MAE {v['mae']:5.2f} m  bias {v['bias']:+5.2f} m  n={v['n']}")
    a = r['lane']
    L.append(f"    LANE   MAE {n(a['mae'])} m   bias {n(a['bias'],'%+.2f')} m"
             f"   p95 {n(a['p95'])} m   coverage {a['coverage']*100:.0f}%")
    s = r['stability']
    L.append(f"    STAB   cte jitter {s['cte_jitter_mean']:.4f} m/frame"
             f"  p99 {s['cte_jitter_p99']:.3f}   |cte| {s['cte_abs_mean']:.2f} m"
             f"   >1m {s['frac_over_1m']*100:.1f}%")
    return '\n'.join(L)


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    outj = None
    if '--json' in sys.argv:
        outj = sys.argv[sys.argv.index('--json') + 1]
        args = [a for a in args if a != outj]
    if len(args) < 2:
        raise SystemExit(__doc__)
    gt_path, logs = args[0], args[1:]
    res = []
    print()
    for lg in logs:
        r = score(gt_path, lg)
        res.append(r)
        print(fmt(r)); print()
    if outj:
        json.dump(res, open(outj, 'w'), indent=2)
        print(f'  wrote {outj}')

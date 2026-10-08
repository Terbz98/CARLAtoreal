"""How far each render's VaVAM plan is from the plan VaVAM makes on raw CARLA at the same moment.
The user's target: the sunny render should drive the policy the same way CARLA does (0 = identical).
Noise floor: VaVAM is not bit-deterministic on CPU; the same input moves a waypoint by up to ~0.1 m."""
import numpy as np, json, glob, os
os.chdir(os.environ.get('ALPASIM_DIR', os.path.dirname(os.path.abspath(__file__))))
out = {}
TOWNS = [t for t in ['town05', 'town10hd', 'town03', 'town04', 'town06', 'town10hd_night'] if os.path.exists(f'results/{t}_CARLA_plans.npy')]
for town in TOWNS:
    def L(t):
        a = np.load(f'results/{town}_{t}_plans.npy'); return {int(r[0]): r[1:].reshape(-1, 2) for r in a}
    C = L('CARLA'); out[town] = {}
    print(f'\n{town}: distance of each render plan from the CARLA plan (m), mean over {len(C)} decisions')
    print(f'{"render":8s} {"1 s":>6s} {"2 s":>6s} {"3 s":>6s} {"lat 3 s":>8s} {"whole path":>11s}')
    tags = sorted(os.path.basename(f)[len(town) + 1:-len('_plans.npy')] for f in glob.glob(f'results/{town}_*_plans.npy'))
    for t in [t for t in tags if t != 'CARLA' and not t.startswith('night_')]:
        R = L(t); ks = [k for k in C if k in R]
        d = np.array([np.linalg.norm(R[k] - C[k], axis=1) for k in ks])
        lat = float(np.mean([abs(R[k][5][1] - C[k][5][1]) for k in ks]))
        out[town][t] = {'d1': d[:, 1].mean(), 'd2': d[:, 3].mean(), 'd3': d[:, 5].mean(), 'lat3': lat, 'path': d.mean()}
        print(f'{t:8s} {d[:,1].mean():6.2f} {d[:,3].mean():6.2f} {d[:,5].mean():6.2f} {lat:8.2f} {d.mean():11.2f}')
json.dump(out, open('results/plan_agreement.json', 'w'), indent=1)

# one line per render across towns: the number to rank by is the mean over the towns it has
tags = sorted({t for tw in out.values() for t in tw})
print(f'\nwhole-path distance from the CARLA plan, by town (m)')
print(f'{"render":9s}' + ''.join(f'{t:>9s}' for t in TOWNS) + f'{"mean":>8s}{"n":>3s}')
for t in tags:
    v = [out[tw][t]['path'] if t in out[tw] else None for tw in TOWNS]
    have = [x for x in v if x is not None]
    print(f'{t:9s}' + ''.join(f'{x:9.2f}' if x is not None else f'{"-":>9s}' for x in v) + f'{np.mean(have):8.2f}{len(have):3d}')
